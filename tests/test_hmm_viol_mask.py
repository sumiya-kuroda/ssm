# Test GLM-HMM with a mask for e.g. violation trials. Confirm that these trials
# have no effect on final loglikelihood

import ssm
import autograd.numpy as np
import autograd.numpy.random as npr

npr.seed(65)

def create_violation_mask(violation_idx, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1
    = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in violation_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonviolation_idx) + len(
        violation_idx
    ) == T, "violation and non-violation idx do not include all dta!"
    return nonviolation_idx, np.expand_dims(mask, axis = 1)

def test_viol_mask():
    # instantiate HMM
    K, D, M, C = 2, 1, 1, 2
    prior_sigma, transition_alpha = 2, 2
    y = npr.choice([-1,0,1], p = [0.06, 0.47, 0.47], size = (100,1))
    inpt = npr.choice([-1, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125,
                          0.25, 1], size = (100, 1))
    # Identify violations for exclusion and create mask
    violation_idx = np.where(y == -1)[0]
    nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                   inpt.shape[0])
    this_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=C,
                                               prior_sigma=prior_sigma),
                       transitions="sticky",
                       transition_kwargs=dict(alpha=transition_alpha,
                                              kappa=0))
    # Replace value of y at violation trials  - this should not affect
    # marginal likelihood (since violation trials should be excluded in
    # calculation
    y_prime = np.copy(y)
    y_prime[np.where(y_prime == -1)] = 1

    y_prime_prime = np.copy(y)
    y_prime_prime[np.where(y_prime_prime == -1)] = 0

    log_prob_1 = this_hmm.log_probability([y_prime],
                       inputs=[inpt],
                       masks=[mask])

    log_prob_2 = this_hmm.log_probability([y_prime_prime],
                                          inputs=[inpt],
                                          masks=[mask])
    assert log_prob_1 == log_prob_2

    # now include set all masks to 1.  full data LL should be less likely
    # when we include all trials vs excluding violation trials
    log_prob_3 = this_hmm.log_probability([y_prime_prime],
                                          inputs=[inpt],
                                          masks=[np.ones(inpt.shape)])
    assert log_prob_3 < log_prob_2