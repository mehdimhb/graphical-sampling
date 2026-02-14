import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter

from ..population import Population


class Moran:
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

            ro.r("""
            library(Matrix)
            library(WaveSampling)
            library(sampling)
            library(BalancedSampling)      
            """)

            # Precompute W once
            ro.r("""
                    W0 <- wpik(coords, probs)
                    W <- W0 - diag(diag(W0))
                    diag(W) <- 0
                """)
            ro.globalenv['samples'] = r_sample_list

            # Define an R function that loops over all samples
            ro.r("""
                    score_moran <- function(W, probs, coords, samples_list) {
                      S <- length(samples_list)
                      IBs   <- numeric(S)

                      for (i in seq_len(S)) {
                        samp_idx <- samples_list[[i]]
                        mask <- integer(length(probs))
                        mask[samp_idx] <- 1

                        IBs[i]   <- tryCatch(IB(W, mask), error = function(e) Inf)
                      }
    
                      cbind(IB = IBs)
                    }
                """)

            # Call it once
            result = ro.r("score_moran(W, probs, coords, samples)")
            # result comes back as an R matrix  S×2

        # Turn it into an (S×2) numpy array
        with localconverter(default_converter + numpy2ri.converter):
            moran_scores = np.array(result)
        return moran_scores.reshape(-1)
