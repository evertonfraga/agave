use {
    crate::{
        error::BlsError,
        hash::{HashedMessage, HashedPoPPayload, PreparedHashedMessage},
        prepared_g2::PreparedG2,
        proof_of_possession::{AsProofOfPossessionAffine, ProofOfPossessionAffine},
        pubkey::points::{AsPubkeyAffine, PopVerified, PubkeyAffine},
        signature::{AsSignatureAffine, SignatureAffine},
    },
    blstrs::G1Affine,
    group::prime::PrimeCurveAffine,
};

/// One of the two Miller loops of a pairing check, as passed to the `join`
/// function of [`VerifySignature::verify_signature_pre_hashed_with_join`] and
/// [`VerifySignature::verify_signature_prepared_with_join`].
///
/// The closure stores its result in the verifier, so calling it again only
/// repeats the same work.
pub type MillerLoop<'a> = &'a mut (dyn FnMut() + Send + 'a);

/// A trait that provides Proof of Possession verification methods to any
/// convertible public key type.
pub trait VerifyPop: AsPubkeyAffine + Sized {
    /// Uses this public key to verify any convertible proof of possession type.
    fn verify_proof_of_possession<P: AsProofOfPossessionAffine>(
        &self,
        proof: &P,
        payload: Option<&[u8]>,
    ) -> Result<(), BlsError> {
        let pubkey_bytes = self.try_as_affine()?.to_bytes_compressed();
        let hashed_pubkey = HashedPoPPayload::new(payload.unwrap_or(&[]), &pubkey_bytes);
        self.verify_proof_of_possession_pre_hashed(proof, &hashed_pubkey)
    }

    /// Uses this public key to verify any convertible proof of possession type.
    fn verify_proof_of_possession_pre_hashed<P: AsProofOfPossessionAffine>(
        &self,
        proof: &P,
        hashed_payload: &HashedPoPPayload,
    ) -> Result<(), BlsError> {
        let pubkey_affine = self.try_as_affine()?;
        let proof_affine = proof.try_as_affine()?;
        pubkey_affine
            ._verify_proof_of_possession(&proof_affine, hashed_payload)
            .then_some(())
            .ok_or(BlsError::VerificationFailed)
    }

    /// Verifies the proof of possession and, upon success, returns a `PopVerified`
    /// wrapper.
    fn verify_and_wrap_pop<P: AsProofOfPossessionAffine>(
        self,
        proof: &P,
        payload: Option<&[u8]>,
    ) -> Result<PopVerified<Self>, BlsError> {
        self.verify_proof_of_possession(proof, payload)?;
        Ok(PopVerified(self))
    }
}

// Blanket implementation so any raw key can attempt to prove itself
impl<T: AsPubkeyAffine> VerifyPop for T {}

/// A trait that provides signature verification methods to public-key-like types.
///
/// The crate-provided implementations are intentionally limited to `PopVerified<T>` and
/// aggregate public-key wrappers, so raw public key types from this crate do not gain
/// direct signature-verification methods before PoP verification. This trait remains public
/// as an extension point for downstream key wrappers. Implementing it for an unverified key
/// type opts out of the crate's PoP-verified usage convention and can reintroduce rogue-key
/// risks in aggregation or signer-attribution flows.
pub trait VerifySignature: AsPubkeyAffine {
    /// Uses this public key to verify any convertible signature type.
    fn verify_signature<S: AsSignatureAffine>(
        &self,
        signature: &S,
        message: &[u8],
    ) -> Result<(), BlsError> {
        let hashed_message = HashedMessage::new(message);
        self.verify_signature_pre_hashed(signature, &hashed_message)
    }

    /// Uses this public key to verify any convertible signature type using a pre-hashed message.
    fn verify_signature_pre_hashed<S: AsSignatureAffine>(
        &self,
        signature: &S,
        hashed_message: &HashedMessage,
    ) -> Result<(), BlsError> {
        let prepared_hashed_message = PreparedHashedMessage::from_hashed_message(hashed_message);
        self.verify_signature_prepared(signature, &prepared_hashed_message)
    }

    /// Uses this public key to verify any convertible signature type using a prepared message.
    fn verify_signature_prepared<S: AsSignatureAffine>(
        &self,
        signature: &S,
        prepared_hashed_message: &PreparedHashedMessage,
    ) -> Result<(), BlsError> {
        let pubkey_affine = self.try_as_affine()?;
        let signature_affine = signature.try_as_affine()?;
        pubkey_affine
            ._verify_signature_prepared(&signature_affine, &prepared_hashed_message.prepared)
            .then_some(())
            .ok_or(BlsError::VerificationFailed)
    }

    /// Uses this public key to verify any convertible signature type using a
    /// pre-hashed message, and returns the same result as
    /// [`Self::verify_signature_pre_hashed`]. The two Miller loops of the
    /// pairing check run through `join`.
    ///
    /// `join` receives the two Miller loops and must run both of them before
    /// it returns, in either order and on any thread. For example,
    /// `|a, b| rayon::join(a, b)` runs them in parallel and `|a, b| (a(), b())`
    /// runs them one after the other. Each Miller loop stores its own result,
    /// so `join` cannot change the outcome, and its return value is ignored.
    /// The conversions of the public key and the signature and the identity
    /// check of the public key happen before `join` is called.
    ///
    /// # Panics
    ///
    /// Panics if `join` returns without running both Miller loops.
    fn verify_signature_pre_hashed_with_join<S, J, R>(
        &self,
        signature: &S,
        hashed_message: &HashedMessage,
        join: J,
    ) -> Result<(), BlsError>
    where
        S: AsSignatureAffine,
        J: FnOnce(MillerLoop<'_>, MillerLoop<'_>) -> R,
    {
        let pubkey_affine = self.try_as_affine()?;
        let signature_affine = signature.try_as_affine()?;
        pubkey_affine
            ._verify_signature_with_join(&signature_affine, hashed_message, join)
            .then_some(())
            .ok_or(BlsError::VerificationFailed)
    }

    /// Uses this public key to verify any convertible signature type using a
    /// prepared message, and returns the same result as
    /// [`Self::verify_signature_prepared`]. The two Miller loops of the
    /// pairing check run through `join`, as in
    /// [`Self::verify_signature_pre_hashed_with_join`].
    ///
    /// # Panics
    ///
    /// Panics if `join` returns without running both Miller loops.
    fn verify_signature_prepared_with_join<S, J, R>(
        &self,
        signature: &S,
        prepared_hashed_message: &PreparedHashedMessage,
        join: J,
    ) -> Result<(), BlsError>
    where
        S: AsSignatureAffine,
        J: FnOnce(MillerLoop<'_>, MillerLoop<'_>) -> R,
    {
        let pubkey_affine = self.try_as_affine()?;
        let signature_affine = signature.try_as_affine()?;
        pubkey_affine
            ._verify_signature_prepared_with_join(
                &signature_affine,
                &prepared_hashed_message.prepared,
                join,
            )
            .then_some(())
            .ok_or(BlsError::VerificationFailed)
    }
}

impl PubkeyAffine {
    /// Verify a signature and a message against a public key
    pub(crate) fn _verify_signature(
        &self,
        signature: &SignatureAffine,
        hashed_message: &HashedMessage,
    ) -> bool {
        if bool::from(self.0.is_identity()) {
            return false;
        }

        // Check e(pubkey, H(m)) = e(g1, signature) without preparing
        // either G2 point.
        let generator = G1Affine::generator();
        let message_pairing =
            blst::blst_fp12::miller_loop(hashed_message.0.as_ref(), self.0.as_ref());
        let signature_pairing =
            blst::blst_fp12::miller_loop(signature.0.as_ref(), generator.as_ref());

        // Compare the pairings using one final exponentiation.
        blst::blst_fp12::finalverify(&message_pairing, &signature_pairing)
    }

    pub(crate) fn _verify_signature_prepared(
        &self,
        signature: &SignatureAffine,
        hashed_message_prepared: &PreparedG2,
    ) -> bool {
        if bool::from(self.0.is_identity()) {
            return false;
        }

        // Reuse the message's prepared table.
        let message_pairing = hashed_message_prepared.miller_loop(&self.0);

        // Compute the signature term directly, without a preparation table.
        let generator = G1Affine::generator();
        let signature_pairing =
            blst::blst_fp12::miller_loop(signature.0.as_ref(), generator.as_ref());

        // Check e(pubkey, H(m)) = e(g1, signature) with one final exponentiation.
        blst::blst_fp12::finalverify(&message_pairing, &signature_pairing)
    }

    /// Computes the same check as `_verify_signature`, with the two Miller
    /// loops run by `join`.
    pub(crate) fn _verify_signature_with_join<J, R>(
        &self,
        signature: &SignatureAffine,
        hashed_message: &HashedMessage,
        join: J,
    ) -> bool
    where
        J: FnOnce(MillerLoop<'_>, MillerLoop<'_>) -> R,
    {
        if bool::from(self.0.is_identity()) {
            return false;
        }

        let generator = G1Affine::generator();
        let mut message_pairing = None;
        let mut signature_pairing = None;
        join(
            &mut || {
                message_pairing = Some(blst::blst_fp12::miller_loop(
                    hashed_message.0.as_ref(),
                    self.0.as_ref(),
                ));
            },
            &mut || {
                signature_pairing = Some(blst::blst_fp12::miller_loop(
                    signature.0.as_ref(),
                    generator.as_ref(),
                ));
            },
        );
        finalverify_joined(message_pairing, signature_pairing)
    }

    /// Computes the same check as `_verify_signature_prepared`, with the two
    /// Miller loops run by `join`.
    pub(crate) fn _verify_signature_prepared_with_join<J, R>(
        &self,
        signature: &SignatureAffine,
        hashed_message_prepared: &PreparedG2,
        join: J,
    ) -> bool
    where
        J: FnOnce(MillerLoop<'_>, MillerLoop<'_>) -> R,
    {
        if bool::from(self.0.is_identity()) {
            return false;
        }

        let generator = G1Affine::generator();
        let mut message_pairing = None;
        let mut signature_pairing = None;
        join(
            &mut || {
                message_pairing = Some(hashed_message_prepared.miller_loop(&self.0));
            },
            &mut || {
                signature_pairing = Some(blst::blst_fp12::miller_loop(
                    signature.0.as_ref(),
                    generator.as_ref(),
                ));
            },
        );
        finalverify_joined(message_pairing, signature_pairing)
    }

    /// Verify a proof of possession against a public key
    pub(crate) fn _verify_proof_of_possession(
        &self,
        proof: &ProofOfPossessionAffine,
        hashed_payload: &HashedPoPPayload,
    ) -> bool {
        if bool::from(self.0.is_identity()) {
            return false;
        }

        // Check e(pubkey, H(payload || pubkey_bytes)) = e(g1, proof)
        // without preparing either G2 point.
        let generator = G1Affine::generator();
        let payload_pairing =
            blst::blst_fp12::miller_loop(hashed_payload.0.as_ref(), self.0.as_ref());
        let proof_pairing = blst::blst_fp12::miller_loop(proof.0.as_ref(), generator.as_ref());

        // Compare the pairings using one final exponentiation.
        blst::blst_fp12::finalverify(&payload_pairing, &proof_pairing)
    }
}

/// Compares the results of the two Miller loops that a join function ran,
/// using one final exponentiation.
fn finalverify_joined(
    message_pairing: Option<blst::blst_fp12>,
    signature_pairing: Option<blst::blst_fp12>,
) -> bool {
    match (message_pairing, signature_pairing) {
        (Some(message_pairing), Some(signature_pairing)) => {
            // Check e(pubkey, H(m)) = e(g1, signature).
            blst::blst_fp12::finalverify(&message_pairing, &signature_pairing)
        }
        _ => panic!("the join function returned without running both Miller loops"),
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::{
            keypair::Keypair,
            pubkey::{PubkeyCompressed, PubkeyProjective},
            signature::{SignatureCompressed, SignatureProjective},
        },
        blstrs::G2Affine,
        std::{thread, vec::Vec},
    };

    type Join = fn(MillerLoop<'_>, MillerLoop<'_>);

    fn sequential(a: MillerLoop<'_>, b: MillerLoop<'_>) {
        a();
        b();
    }

    fn reversed(a: MillerLoop<'_>, b: MillerLoop<'_>) {
        b();
        a();
    }

    fn repeated(a: MillerLoop<'_>, b: MillerLoop<'_>) {
        a();
        b();
        b();
        a();
    }

    fn scoped_thread(a: MillerLoop<'_>, b: MillerLoop<'_>) {
        thread::scope(|scope| {
            scope.spawn(a);
            b();
        });
    }

    #[cfg(feature = "parallel")]
    fn rayon_join(a: MillerLoop<'_>, b: MillerLoop<'_>) {
        rayon::join(a, b);
    }

    fn joins() -> Vec<(&'static str, Join)> {
        #[allow(unused_mut)]
        let mut joins: Vec<(&'static str, Join)> = std::vec![
            ("sequential", sequential),
            ("reversed", reversed),
            ("repeated", repeated),
            ("scoped thread", scoped_thread),
        ];
        #[cfg(feature = "parallel")]
        joins.push(("rayon", rayon_join));
        joins
    }

    /// Checks that the calls with a join function return the same result as
    /// the calls without one, and returns that result.
    fn check_joined<P: VerifySignature, S: AsSignatureAffine>(
        label: &str,
        pubkey: &P,
        signature: &S,
        hashed_message: &HashedMessage,
    ) -> Result<(), BlsError> {
        let prepared_hashed_message = PreparedHashedMessage::from_hashed_message(hashed_message);
        let expected = pubkey.verify_signature_pre_hashed(signature, hashed_message);
        let expected_prepared =
            pubkey.verify_signature_prepared(signature, &prepared_hashed_message);
        for (join_label, join) in joins() {
            assert_eq!(
                pubkey.verify_signature_pre_hashed_with_join(signature, hashed_message, join),
                expected,
                "{label}: pre-hashed, {join_label} join",
            );
            assert_eq!(
                pubkey.verify_signature_prepared_with_join(
                    signature,
                    &prepared_hashed_message,
                    join,
                ),
                expected_prepared,
                "{label}: prepared, {join_label} join",
            );
        }
        expected
    }

    #[test]
    fn test_joined_verification_matches_sequential_verification() {
        let keypairs = [
            Keypair::derive(&[50u8; 32]).unwrap(),
            Keypair::derive(&[51u8; 32]).unwrap(),
        ];
        let messages: [&[u8]; 2] = [b"joined verification", b""];
        let hashes = messages.map(HashedMessage::new);

        for (signer_index, signer) in keypairs.iter().enumerate() {
            for (signed_index, message) in messages.iter().enumerate() {
                let signature = signer.sign(message);
                let compressed = SignatureCompressed::from(signature);
                for (verifier_index, verifier) in keypairs.iter().enumerate() {
                    for (checked_index, hashed_message) in hashes.iter().enumerate() {
                        let expected =
                            if signer_index == verifier_index && signed_index == checked_index {
                                Ok(())
                            } else {
                                Err(BlsError::VerificationFailed)
                            };
                        let label = std::format!(
                            "signer {signer_index}, message {signed_index}, verifier \
                             {verifier_index}, checked message {checked_index}"
                        );
                        assert_eq!(
                            check_joined(&label, &verifier.public, &signature, hashed_message),
                            expected,
                            "{label}",
                        );
                        assert_eq!(
                            check_joined(&label, &verifier.public, &compressed, hashed_message),
                            expected,
                            "{label}, compressed signature",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_joined_verification_of_aggregates() {
        let keypairs = [
            Keypair::derive(&[52u8; 32]).unwrap(),
            Keypair::derive(&[53u8; 32]).unwrap(),
            Keypair::derive(&[54u8; 32]).unwrap(),
        ];
        let message = b"aggregate";
        let hashed_message = HashedMessage::new(message);
        let aggregate_pubkey =
            PubkeyProjective::aggregate(keypairs.iter().map(|keypair| &keypair.public)).unwrap();
        // The aggregate as agave builds it from PoP-verified keys.
        let pop_verified_aggregate = unsafe { PopVerified::new_unchecked(*aggregate_pubkey) };

        let signatures: Vec<_> = keypairs
            .iter()
            .map(|keypair| keypair.sign(message))
            .collect();
        let aggregate_signature = SignatureProjective::aggregate(signatures.iter()).unwrap();
        let mut wrong_signatures = signatures.clone();
        wrong_signatures[1] = keypairs[1].sign(b"another message");
        let wrong_aggregate_signature =
            SignatureProjective::aggregate(wrong_signatures.iter()).unwrap();

        for (label, signature, expected) in [
            ("valid aggregate", aggregate_signature, Ok(())),
            (
                "aggregate with a wrong signature",
                wrong_aggregate_signature,
                Err(BlsError::VerificationFailed),
            ),
        ] {
            assert_eq!(
                check_joined(label, &aggregate_pubkey, &signature, &hashed_message),
                expected,
                "{label}",
            );
            assert_eq!(
                check_joined(label, &pop_verified_aggregate, &signature, &hashed_message),
                expected,
                "{label}, PoP-verified aggregate public key",
            );
        }
    }

    #[test]
    fn test_joined_verification_of_identity_and_malformed_inputs() {
        let keypair = Keypair::derive(&[55u8; 32]).unwrap();
        let message = b"identity and malformed inputs";
        let hashed_message = HashedMessage::new(message);
        let identity_hashed_message = HashedMessage(G2Affine::identity());

        let valid_signature = SignatureCompressed::from(keypair.sign(message));
        let identity_signature = SignatureCompressed::from(SignatureProjective::identity());
        let malformed_signature = SignatureCompressed([0u8; 96]);
        let signature_decode_error = SignatureAffine::try_from(&malformed_signature).unwrap_err();

        let identity_pubkey =
            unsafe { PopVerified::new_unchecked(PubkeyAffine::from(PubkeyProjective::identity())) };
        let malformed_pubkey = unsafe { PopVerified::new_unchecked(PubkeyCompressed([0u8; 48])) };
        let pubkey_decode_error = malformed_pubkey.try_as_affine().unwrap_err();

        for (label, signature, expected) in [
            ("valid signature", valid_signature, Ok(())),
            (
                "identity signature",
                identity_signature,
                Err(BlsError::VerificationFailed),
            ),
            (
                "malformed signature",
                malformed_signature,
                Err(signature_decode_error.clone()),
            ),
        ] {
            assert_eq!(
                check_joined(label, &keypair.public, &signature, &hashed_message),
                expected,
                "{label}",
            );
        }

        // The identity public key is rejected after both conversions.
        for (label, signature, expected) in [
            (
                "identity public key, valid signature",
                valid_signature,
                Err(BlsError::VerificationFailed),
            ),
            (
                "identity public key, malformed signature",
                malformed_signature,
                Err(signature_decode_error),
            ),
        ] {
            assert_eq!(
                check_joined(label, &identity_pubkey, &signature, &hashed_message),
                expected,
                "{label}",
            );
        }
        assert_eq!(
            check_joined(
                "malformed public key",
                &malformed_pubkey,
                &valid_signature,
                &hashed_message,
            ),
            Err(pubkey_decode_error),
        );

        // An identity message cannot come from hashing, but both kinds of
        // calls must still agree on it.
        for (label, signature) in [
            ("identity message, valid signature", valid_signature),
            ("identity message, identity signature", identity_signature),
        ] {
            check_joined(label, &keypair.public, &signature, &identity_hashed_message).ok();
        }
    }

    #[test]
    fn test_join_is_not_called_for_rejected_inputs() {
        fn must_not_run(_: MillerLoop<'_>, _: MillerLoop<'_>) {
            panic!("join must not be called");
        }

        let keypair = Keypair::derive(&[56u8; 32]).unwrap();
        let hashed_message = HashedMessage::new(b"rejected inputs");
        let prepared_hashed_message = PreparedHashedMessage::from_hashed_message(&hashed_message);
        let valid_signature = SignatureCompressed::from(keypair.sign(b"rejected inputs"));
        let malformed_signature = SignatureCompressed([0u8; 96]);
        let identity_pubkey =
            unsafe { PopVerified::new_unchecked(PubkeyAffine::from(PubkeyProjective::identity())) };
        let malformed_pubkey = unsafe { PopVerified::new_unchecked(PubkeyCompressed([0u8; 48])) };

        assert!(keypair
            .public
            .verify_signature_pre_hashed_with_join(
                &malformed_signature,
                &hashed_message,
                must_not_run
            )
            .is_err());
        assert!(keypair
            .public
            .verify_signature_prepared_with_join(
                &malformed_signature,
                &prepared_hashed_message,
                must_not_run,
            )
            .is_err());
        assert!(identity_pubkey
            .verify_signature_pre_hashed_with_join(&valid_signature, &hashed_message, must_not_run)
            .is_err());
        assert!(identity_pubkey
            .verify_signature_prepared_with_join(
                &valid_signature,
                &prepared_hashed_message,
                must_not_run,
            )
            .is_err());
        assert!(malformed_pubkey
            .verify_signature_pre_hashed_with_join(&valid_signature, &hashed_message, must_not_run)
            .is_err());
        assert!(malformed_pubkey
            .verify_signature_prepared_with_join(
                &valid_signature,
                &prepared_hashed_message,
                must_not_run,
            )
            .is_err());
    }

    #[test]
    #[should_panic(expected = "the join function returned without running both Miller loops")]
    fn test_join_that_skips_a_miller_loop_panics() {
        let keypair = Keypair::derive(&[57u8; 32]).unwrap();
        let hashed_message = HashedMessage::new(b"skipped Miller loop");
        let signature = keypair.sign(b"skipped Miller loop");
        let _ = keypair.public.verify_signature_pre_hashed_with_join(
            &signature,
            &hashed_message,
            |a: MillerLoop<'_>, _: MillerLoop<'_>| a(),
        );
    }
}
