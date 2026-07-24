use hdf5_pure::{Error, File, FormatError, VlenStringReadOptions};
use tempfile::tempdir;

const FIXTURE: &str = "tests/fixtures/vl_strings.h5";

fn expected_names() -> Vec<String> {
    ["Alice", "Bob", "Charlie"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

#[test]
fn buffered_vlen_dataset_read_and_size() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();

    assert_eq!(dataset.vlen_string_payload_size().unwrap(), 15);
    assert_eq!(dataset.read_string().unwrap(), expected_names());
    assert_eq!(
        dataset
            .read_vlen_strings(
                VlenStringReadOptions::new()
                    .with_max_elements(3)
                    .with_max_payload_bytes(15),
            )
            .unwrap(),
        expected_names()
    );
}

#[test]
fn vlen_element_limit_is_checked_before_payload_read() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let error = dataset
        .read_vlen_strings(VlenStringReadOptions::new().with_max_elements(2))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Format(FormatError::VariableLengthElementLimitExceeded {
            limit: 2,
            actual: 3,
        })
    ));
}

#[test]
fn vlen_payload_limit_reports_required_bytes() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let error = dataset
        .read_vlen_strings(VlenStringReadOptions::new().with_max_payload_bytes(14))
        .unwrap_err();

    assert!(matches!(
        error,
        Error::Format(FormatError::VariableLengthByteLimitExceeded {
            limit: 14,
            required: 15,
        })
    ));
}

#[test]
fn vlen_visitor_delivers_strings_in_order() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let mut names = Vec::new();

    dataset
        .visit_vlen_strings(
            VlenStringReadOptions::new().with_max_payload_bytes(15),
            |name| names.push(name.to_owned()),
        )
        .unwrap();

    assert_eq!(names, expected_names());
}

#[test]
fn vlen_specific_apis_reject_non_vlen_datasets() {
    let file = File::open(FIXTURE).unwrap();
    let dataset = file.dataset("names").unwrap();
    let datatype = dataset.datatype().unwrap();
    assert!(matches!(
        datatype,
        hdf5_pure::Datatype::VariableLength { .. }
    ));

    let numeric = File::open("tests/fixtures/simple_dataset.h5").unwrap();
    let numeric_dataset = numeric.dataset("data").unwrap();
    assert!(matches!(
        numeric_dataset.vlen_string_payload_size(),
        Err(Error::Format(FormatError::TypeMismatch { .. }))
    ));
}

// --- One-collection heap object limit (u16 index; issue: silent corruption) ---

/// Exactly `u16::MAX` strings is the most one collection can index; the file
/// must round-trip. One past it must be refused at write time — previously the
/// on-disk object index wrapped (the 65,536th object's header read as the
/// free-space marker) and the file's references could not be resolved by this
/// crate or the C library.
#[test]
fn heap_object_limit_boundary_roundtrips() {
    let values: Vec<String> = (0..u16::MAX as usize).map(|i| format!("s{i}")).collect();
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&refs);
    let bytes = builder.finish().unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let read = file.dataset("labels").unwrap().read_string().unwrap();
    assert_eq!(read.len(), u16::MAX as usize);
    assert_eq!(read.first().map(String::as_str), Some("s0"));
    assert_eq!(read.last().map(String::as_str), Some("s65534"));
}

#[test]
fn heap_object_limit_refused_for_datasets() {
    let values: Vec<String> = (0..=u16::MAX as usize).map(|i| format!("s{i}")).collect();
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.create_dataset("labels").with_vlen_strings(&refs);
    let error = builder.finish().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::GlobalHeapObjectLimitExceeded { count: 65_536 })
        ),
        "expected GlobalHeapObjectLimitExceeded, got {error:?}"
    );
}

#[test]
fn heap_object_limit_refused_for_attributes() {
    let values: Vec<String> = (0..=u16::MAX as usize).map(|i| format!("s{i}")).collect();
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.set_attr("labels", hdf5_pure::AttrValue::VarLenAsciiArray(values));
    builder.create_dataset("x").with_f64_data(&[1.0]);
    let error = builder.finish().unwrap_err();
    assert!(
        matches!(
            error,
            Error::Format(FormatError::GlobalHeapObjectLimitExceeded { count: 65_536 })
        ),
        "expected GlobalHeapObjectLimitExceeded, got {error:?}"
    );
}

/// Strings one past the limit, as the shared over-limit payload for the edit
/// session cases below.
fn over_limit_strings() -> Vec<String> {
    (0..=u16::MAX as usize).map(|i| format!("s{i}")).collect()
}

/// Assert a staged edit was refused with the typed limit error, whether it
/// surfaced when staging or on commit.
#[track_caller]
fn assert_over_limit(result: Result<(), Error>) {
    let error = result.expect_err("expected the over-limit write to be refused");
    assert!(
        matches!(
            error,
            Error::Format(FormatError::GlobalHeapObjectLimitExceeded { count: 65_536 })
        ),
        "expected GlobalHeapObjectLimitExceeded, got {error:?}"
    );
}

/// A starter file for the edit-session cases: one small dataset, nothing
/// variable-length.
fn write_starter(path: &std::path::Path) {
    let mut builder = hdf5_pure::FileBuilder::new();
    builder.create_dataset("d").with_f64_data(&[1.0]);
    builder.write(path).unwrap();
}

/// The edit session stages a dataset through a different pipeline than
/// `FileBuilder` (`flatten_dataset`), so it needs its own refusal: adding an
/// over-limit variable-length dataset to an existing file must fail on commit.
#[test]
fn heap_object_limit_refused_by_edit_session_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("d.h5");
    write_starter(&path);

    let values = over_limit_strings();
    let refs: Vec<&str> = values.iter().map(String::as_str).collect();
    let file = File::open_rw(&path).unwrap();
    file.root()
        .create_dataset("labels", |b| {
            b.with_vlen_strings(&refs);
        })
        .unwrap();
    assert_over_limit(file.commit());
}

/// Same pipeline, the attribute half: a variable-length attribute carried on a
/// dataset staged by the edit session.
#[test]
fn heap_object_limit_refused_by_edit_session_dataset_attr() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("da.h5");
    write_starter(&path);

    let values = over_limit_strings();
    let file = File::open_rw(&path).unwrap();
    file.root()
        .create_dataset("x", |b| {
            b.with_f64_data(&[1.0])
                .set_attr("labels", hdf5_pure::AttrValue::VarLenAsciiArray(values));
        })
        .unwrap();
    assert_over_limit(file.commit());
}

/// The in-place attribute path (`apply_group_attr_ops`) is a third pipeline,
/// and it refuses an over-limit variable-length attribute *earlier*: a compact
/// attribute message is capped at 65,535 serialized bytes, and 16 bytes of
/// reference per element means that cap binds long before the heap-object one.
/// Pinned here so the ordering is deliberate — the heap-object guard behind it
/// only becomes load-bearing if the compact cap lifts (dense attribute storage,
/// [#102](https://github.com/stephenberry/hdf5-pure/issues/102)).
#[test]
fn in_place_group_attr_refuses_over_limit_by_message_size() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ga.h5");
    write_starter(&path);

    let values = over_limit_strings();
    let file = File::open_rw(&path).unwrap();
    let error = file
        .root()
        .set_attr("labels", hdf5_pure::AttrValue::VarLenAsciiArray(values))
        .and_then(|()| file.commit())
        .expect_err("expected the over-limit attribute to be refused");
    match error {
        Error::EditUnsupported(msg) => assert!(
            msg.contains("too large"),
            "expected the compact-size refusal, got {msg:?}"
        ),
        other => panic!("expected EditUnsupported, got {other:?}"),
    }
}
