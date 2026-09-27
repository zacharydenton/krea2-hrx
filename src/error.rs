//! The crate's one error type.
//!
//! Every module returns [`Error`], so failures cross layers without being
//! flattened to strings: a runtime failure keeps its [`hrx::Error`] kind
//! (device loss, a busy slot) all the way to the caller, and the caller can
//! tell its own mistakes from the machine's with [`Error::is_invalid_argument`].

/// What went wrong, classified by who can fix it.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The request cannot be served as asked: dimensions, arguments, a prompt
    /// or a model file the caller named. Retrying unchanged will fail again.
    #[error("{0}")]
    InvalidArgument(String),
    /// The progress callback asked to abandon the image.
    #[error("generation cancelled")]
    Cancelled,
    /// A panic during an earlier call left this object in an unknown state.
    /// Create a new one.
    #[error("the {0} is poisoned by an earlier panic; create a new one")]
    Poisoned(&'static str),
    /// The GPU runtime, compiler or driver failed, or refused the work.
    #[error(transparent)]
    Runtime(#[from] hrx::Error),
    /// The host broke its own contract with a kernel or buffer. A bug here.
    #[error("{0}")]
    Internal(String),
}

/// A result carrying [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// A request the caller can correct.
    pub fn invalid(message: impl Into<String>) -> Error {
        Error::InvalidArgument(message.into())
    }

    /// A broken host-side invariant.
    pub fn internal(message: impl Into<String>) -> Error {
        Error::Internal(message.into())
    }

    /// Whether the caller could have prevented this by asking differently.
    pub fn is_invalid_argument(&self) -> bool {
        matches!(self, Error::InvalidArgument(_))
    }
}

/// Back into the runtime's terms, for work that runs inside an HRX callback. A
/// runtime error passes through unchanged, so its kind survives the round trip.
impl From<Error> for hrx::Error {
    fn from(error: Error) -> hrx::Error {
        match error {
            Error::Runtime(error) => error,
            other => hrx::Error::Message(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_runtime_error_keeps_its_kind_through_a_callback_boundary() {
        let lost = Error::from(hrx::Error::DeviceLost("gone".into()));
        assert!(matches!(hrx::Error::from(lost), hrx::Error::DeviceLost(_)));
        let invalid = hrx::Error::from(Error::invalid("bad width"));
        assert!(matches!(invalid, hrx::Error::Message(message) if message == "bad width"));
    }

    #[test]
    fn only_the_callers_mistakes_are_invalid_arguments() {
        assert!(Error::invalid("bad width").is_invalid_argument());
        assert!(!Error::internal("host bug").is_invalid_argument());
        assert!(!Error::Cancelled.is_invalid_argument());
        assert_eq!(Error::Cancelled.to_string(), "generation cancelled");
    }
}
