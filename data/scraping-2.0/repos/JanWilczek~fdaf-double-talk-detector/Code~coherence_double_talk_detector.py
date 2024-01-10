from numpy.fft import fft
from coherence_calculator import CoherenceCalculator


class CoherenceDoubleTalkDetector:
    """
        A double-talk detector based on coherence between loudspeaker and microphone signal (open loop),
        and estimated loudspeaker signal at microphone and microphone signal (closed loop).
    """
    def __init__(self, block_length, lambda_coherence):
        self.block_length = block_length
        self.open_loop_coherence = CoherenceCalculator(block_length, lambda_coherence)
        self.closed_loop_coherence = CoherenceCalculator(block_length, lambda_coherence)
   
    def is_double_talk(self, loudspeaker_samples_block, microphone_samples_block, microphone_samples_estimate):
        """
        Returns
        -------
        open_loop_rho:  number
            decision variable in range [0, 1]. If close to 1 then no double-talk is present. Typically more noisy
            than its closed-loop equivalent.
        closed_loop_rho:  number
            decision variable in range [0, 1]. If close to 1 then no double-talk is present.
        """
        D_b = fft(microphone_samples_block, axis=0)
        X_b = fft(loudspeaker_samples_block, axis=0)
        Y_hat = fft(microphone_samples_estimate, axis=0)

        open_loop_rho = self.open_loop_coherence.calculate_rho(X_b, D_b)
        closed_loop_rho = self.closed_loop_coherence.calculate_rho(Y_hat, D_b)

        return open_loop_rho, closed_loop_rho