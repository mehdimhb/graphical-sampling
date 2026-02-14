import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter

from ..population import Population


class LocalBalance:
    def __init__(self, population: Population):
        self.population = population
        self.coords = self.population.coords
        self.probs = self.population.probs

    def score(self, samples: np.ndarray) -> np.ndarray:
        r_sample_list = ro.ListVector({
            str(i + 1): ro.IntVector(sample.astype(int) + 1)
            for i, sample in enumerate(samples)
        })

        with localconverter(default_converter + numpy2ri.converter):
            ro.globalenv['coords'] = self.coords
            ro.globalenv['probs'] = self.probs
            ro.globalenv['samples'] = r_sample_list

            ro.r("""
                library(Matrix)
                library(WaveSampling)
                library(sampling)
                library(BalancedSampling)      
            """)

            # Define an R function that loops over all samples
            ro.r("""
                    score_local_balance <- function(W, probs, coords, samples_list) {
                      S <- length(samples_list)
                      SBLBs <- numeric(S)

                      for (i in seq_len(S)) {
                        samp_idx <- samples_list[[i]]
                        mask <- integer(length(probs))
                        mask[samp_idx] <- 1

                        SBLBs[i] <- tryCatch(sblb(probs, coords, samp_idx), error = function(e) Inf)
                      }
                      cbind(SBLB = SBLBs)
                    }
                """)

            # Call it once
            result = ro.r("score_local_balance(W, probs, coords, samples)")
            # result comes back as an R matrix  S×2

        # Turn it into an (S×2) numpy array
        with localconverter(default_converter + numpy2ri.converter):
            local_balance_scores = np.array(result)
        return local_balance_scores.reshape(-1)
