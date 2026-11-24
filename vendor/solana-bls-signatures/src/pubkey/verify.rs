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

        let generator = G1Affine::generator();
        // Unlike blst_miller_loop, blst_miller_loop_n has no special case for
        // points at infinity, so an identity signature or message keeps the
        // two separate Miller loops.
        if bool::from(signature.0.is_identity()) || bool::from(hashed_message.0.is_identity()) {
            let message_pairing =
                blst::blst_fp12::miller_loop(hashed_message.0.as_ref(), self.0.as_ref());
            let signature_pairing =
                blst::blst_fp12::miller_loop(signature.0.as_ref(), generator.as_ref());
            return blst::blst_fp12::finalverify(&message_pairing, &signature_pairing);
        }

        // Check e(pubkey, H(m)) * e(-g1, signature) = 1, which is
        // e(pubkey, H(m)) = e(g1, signature), without preparing either G2
        // point. One Miller loop over both pairs squares its accumulator once
        // per step instead of once per pair, and one final exponentiation
        // follows.
        let neg_generator = -generator;
        let q: [*const blst::blst_p2_affine; 2] = [hashed_message.0.as_ref(), signature.0.as_ref()];
        let p: [*const blst::blst_p1_affine; 2] = [self.0.as_ref(), neg_generator.as_ref()];
        let mut pairing = blst::blst_fp12::default();
        // SAFETY: q and p each hold two pointers to valid affine points, none
        // of them at infinity, which is what blst_miller_loop_n requires for
        // n = 2. The points and pairing stay valid for the call, blst writes
        // only to pairing and keeps none of the pointers. Unlike
        // blst_fp12::miller_loop_n, the raw binding runs on this thread.
        unsafe {
            blst::blst_miller_loop_n(&mut pairing, q.as_ptr(), p.as_ptr(), q.len());
        }
        let pairing = pairing.final_exp();
        // SAFETY: pairing is an initialized blst_fp12 that blst only reads.
        unsafe { blst::blst_fp12_is_one(&pairing) }
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

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::{keypair::Keypair, pubkey::PubkeyProjective, signature::SignatureProjective},
        blstrs::G2Affine,
        std::vec::Vec,
    };

    /// The check of `_verify_signature` with one Miller loop per pair, as it
    /// was before the two pairs shared one Miller loop.
    fn verify_signature_with_two_miller_loops(
        pubkey: &PubkeyAffine,
        signature: &SignatureAffine,
        hashed_message: &HashedMessage,
    ) -> bool {
        if bool::from(pubkey.0.is_identity()) {
            return false;
        }
        let generator = G1Affine::generator();
        let message_pairing =
            blst::blst_fp12::miller_loop(hashed_message.0.as_ref(), pubkey.0.as_ref());
        let signature_pairing =
            blst::blst_fp12::miller_loop(signature.0.as_ref(), generator.as_ref());
        blst::blst_fp12::finalverify(&message_pairing, &signature_pairing)
    }

    /// Checks that `_verify_signature` returns the same result as the check
    /// with two Miller loops and, for a message that is not the identity, as
    /// the check with a prepared message. Returns that result.
    fn check(
        label: &str,
        pubkey: &PubkeyAffine,
        signature: &SignatureAffine,
        hashed_message: &HashedMessage,
    ) -> bool {
        let expected = verify_signature_with_two_miller_loops(pubkey, signature, hashed_message);
        assert_eq!(
            pubkey._verify_signature(signature, hashed_message),
            expected,
            "{label}",
        );
        if !bool::from(hashed_message.0.is_identity()) {
            let prepared = PreparedG2::from(hashed_message.0);
            assert_eq!(
                pubkey._verify_signature_prepared(signature, &prepared),
                expected,
                "{label}: prepared",
            );
        }
        expected
    }

    #[test]
    fn test_miller_loop_pair_matches_two_miller_loops() {
        let keypairs = [
            Keypair::derive(&[60u8; 32]).unwrap(),
            Keypair::derive(&[61u8; 32]).unwrap(),
        ];
        let messages: [&[u8]; 2] = [b"miller loop pair", b""];
        let hashes = messages.map(HashedMessage::new);

        for (signer_index, signer) in keypairs.iter().enumerate() {
            for (signed_index, message) in messages.iter().enumerate() {
                let signature = SignatureAffine::from(signer.sign(message));
                for (verifier_index, verifier) in keypairs.iter().enumerate() {
                    for (checked_index, hashed_message) in hashes.iter().enumerate() {
                        let label = std::format!(
                            "signer {signer_index}, message {signed_index}, verifier \
                             {verifier_index}, checked message {checked_index}"
                        );
                        assert_eq!(
                            check(&label, &verifier.public, &signature, hashed_message),
                            signer_index == verifier_index && signed_index == checked_index,
                            "{label}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_miller_loop_pair_matches_two_miller_loops_for_aggregates() {
        let keypairs = [
            Keypair::derive(&[62u8; 32]).unwrap(),
            Keypair::derive(&[63u8; 32]).unwrap(),
            Keypair::derive(&[64u8; 32]).unwrap(),
        ];
        let message = b"aggregate";
        let hashed_message = HashedMessage::new(message);
        let aggregate_pubkey = PubkeyAffine::from(
            *PubkeyProjective::aggregate(keypairs.iter().map(|keypair| &keypair.public)).unwrap(),
        );
        let signatures: Vec<_> = keypairs
            .iter()
            .map(|keypair| keypair.sign(message))
            .collect();
        let aggregate_signature =
            SignatureAffine::from(SignatureProjective::aggregate(signatures.iter()).unwrap());
        let mut wrong_signatures = signatures.clone();
        wrong_signatures[1] = keypairs[1].sign(b"another message");
        let wrong_aggregate_signature =
            SignatureAffine::from(SignatureProjective::aggregate(wrong_signatures.iter()).unwrap());

        assert!(check(
            "valid aggregate",
            &aggregate_pubkey,
            &aggregate_signature,
            &hashed_message,
        ));
        assert!(!check(
            "aggregate with a wrong signature",
            &aggregate_pubkey,
            &wrong_aggregate_signature,
            &hashed_message,
        ));
    }

    #[test]
    fn test_miller_loop_pair_identity_inputs() {
        let keypair = Keypair::derive(&[65u8; 32]).unwrap();
        let message = b"identity inputs";
        let hashed_message = HashedMessage::new(message);
        let identity_hashed_message = HashedMessage(G2Affine::identity());
        let signature = SignatureAffine::from(keypair.sign(message));
        let identity_signature = SignatureAffine::from(SignatureProjective::identity());
        let identity_pubkey = PubkeyAffine::from(PubkeyProjective::identity());

        assert!(!check(
            "identity signature",
            &keypair.public,
            &identity_signature,
            &hashed_message,
        ));
        assert!(!check(
            "identity public key",
            &identity_pubkey,
            &signature,
            &hashed_message,
        ));
        // An identity message cannot come from hashing. Both Miller loops of
        // the identity message and the identity signature give one, so that
        // pair passes the check.
        assert!(!check(
            "identity message",
            &keypair.public,
            &signature,
            &identity_hashed_message,
        ));
        assert!(check(
            "identity message and identity signature",
            &keypair.public,
            &identity_signature,
            &identity_hashed_message,
        ));
    }
}
