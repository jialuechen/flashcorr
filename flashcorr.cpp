/**
 * FlashBaseCorr.cpp
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <iomanip>

// --- Including QuantLib ---
#include <ql/quantlib.hpp>
#include <ql/math/distributions/bivariatenormaldistribution.hpp>
#include <ql/math/distributions/normaldistribution.hpp>

using QuantLib::Real;
using QuantLib::Integer;
using QuantLib::Size;

// Helper: Bivariate PDF for Analytical Gradient
namespace MathHelper {
    const Real PI = 3.14159265358979323846;
    inline Real bivariate_pdf(Real x, Real y, Real rho) {
        Real one_m_rho2 = 1.0 - rho * rho;
        if (one_m_rho2 < 1e-12) one_m_rho2 = 1e-12;
        Real exponent = (x*x - 2.0*rho*x*y + y*y) / one_m_rho2;
        return (1.0 / (2.0 * PI * std::sqrt(one_m_rho2))) * std::exp(-0.5 * exponent);
    }
}

class FlashBaseCorrSolver {
private:
    Real R;
    Real K_prime;
    Real alpha;      // N^-1(PD)
    Real beta;       // N^-1(K')
    Real LGD;        // 1 - R

public:
    FlashBaseCorrSolver(Real cumulative_pd_isda, Real recovery, Real detachment) 
        : R(recovery) {
        
        cumulative_pd_isda = std::max(1e-7, std::min(0.9999, cumulative_pd_isda));
        QuantLib::InverseCumulativeNormal inv_norm;
        
        this->alpha = inv_norm(cumulative_pd_isda);
        this->LGD = 1.0 - R;
        this->K_prime = detachment / LGD;
        
        // Safety clamp for K'
        this->K_prime = std::min(this->K_prime, 0.9999);
        this->beta = inv_norm(this->K_prime);
    }

    struct EvalResult {
        Real value; 
        Real grad;  
        Real hess;  
    };

    // --- Core Engine ---
    // Returns {EL, dEL/drho, d2EL/drho2}
    EvalResult compute_engine(Real rho) const {
        const Real EPS = 1e-5;
        // Clamp rho to avoid singularity at 0 and 1
        Real r_safe = std::max(EPS, std::min(1.0 - EPS, rho));
        
        Real sqrt_r = std::sqrt(r_safe);
        Real sqrt_1_m_r = std::sqrt(1.0 - r_safe);
        Real r_copula = -sqrt_r; // Correlation for Phi2

        // 1. Calculate dynamic d2 (The Critical Fix)
        // d2 = ( beta * sqrt(1-rho) - alpha ) / sqrt(rho)
        Real d2 = (beta * sqrt_1_m_r - alpha) / sqrt_r;

        // 2. Calculate Value
        // EL = LGD * Phi2(alpha, d2; -sqrt(rho)) + K * (1 - N(d2))
        // Note: K in formula is un-normalized detachment amount = K_prime * LGD
        QuantLib::BivariateCumulativeNormalDistributionDr78 drezner(r_copula);
        Real phi2 = drezner(alpha, d2);
        
        // Standard Normal CDF for d2
        // QuantLib's CumulativeNormalDistribution is slightly heavy to construct, 
        // using std::erfc based implementation for speed if possible, but QL is safer.
        static const QuantLib::CumulativeNormalDistribution norm_cdf;
        Real val_vas = norm_cdf(d2);

        Real detachment_amt = K_prime * LGD;
        Real total_val = LGD * phi2 + detachment_amt * (1.0 - val_vas);

        // 3. Calculate Gradient (The Magic Cancellation Fix)
        // dEL/drho = LGD * dPhi2/drho (Partial)
        // The terms involving d(d2)/drho cancel out with d(Vasicek)/drho!
        
        // d(Phi2)/d(rho_cop) = PDF(alpha, d2, rho_cop)
        Real phi2_pdf = MathHelper::bivariate_pdf(alpha, d2, r_copula);
        
        // Chain rule: rho_cop = -sqrt(rho)
        // d(rho_cop)/drho = -0.5 / sqrt(rho)
        Real d_rhocop_drho = -0.5 / sqrt_r;
        
        // Exact Analytical Gradient
        Real total_grad = LGD * phi2_pdf * d_rhocop_drho;

        // 4. Calculate Hessian (The Engineering Fix)
        // Analytical Hessian of the new formula is still complex (involves d(d2)/drho).
        // Finite Differencing the Gradient is O(1) extra cost and very stable.
        // We need grad at rho+h.
        
        Real h = 1e-5;
        // Re-compute grad at rho + h
        Real r_up = std::min(1.0 - EPS, r_safe + h);
        Real sqrt_r_up = std::sqrt(r_up);
        Real r_cop_up = -sqrt_r_up;
        Real d2_up = (beta * std::sqrt(1.0 - r_up) - alpha) / sqrt_r_up;
        Real pdf_up = MathHelper::bivariate_pdf(alpha, d2_up, r_cop_up);
        Real d_rhocop_up = -0.5 / sqrt_r_up;
        Real grad_up = LGD * pdf_up * d_rhocop_up;

        // Central Difference for Hessian
        // We use (grad_up - grad) / h (Forward) or (grad_up - grad_down)/2h
        // Forward diff is sufficient for Halley direction
        Real total_hess = (grad_up - total_grad) / (r_up - r_safe);

        return {total_val, total_grad, total_hess};
    }

    Real solve(Real market_el_val) {
        // --- Smart Guess (Corrected Logic) ---
        // Equity Tranche: EL decreases as Rho increases (typically)
        
        // Probe endpoints
        Real rho_min = 0.05;
        Real rho_max = 0.95;
        Real el_at_min = compute_engine(rho_min).value;
        Real el_at_max = compute_engine(rho_max).value;
        
        Real rho_guess;
        
        // Handle monotonicity check
        Real el_ceil = std::max(el_at_min, el_at_max);
        Real el_floor = std::min(el_at_min, el_at_max);
        
        if (market_el_val >= el_ceil) {
            rho_guess = (el_at_min > el_at_max) ? rho_min : rho_max; 
        } else if (market_el_val <= el_floor) {
            rho_guess = (el_at_min > el_at_max) ? rho_max : rho_min;
        } else {
            // Linear Interpolation
            Real slope = (rho_max - rho_min) / (el_at_max - el_at_min);
            rho_guess = rho_min + (market_el_val - el_at_min) * slope;
        }

        // Z-Transform
        Real z = std::atanh(std::sqrt(rho_guess));
        z = std::max(0.2, std::min(3.0, z));

        // --- Halley Loop ---
        for(Size i=0; i<10; ++i) {
            Real tanh_z = std::tanh(z);
            Real rho = tanh_z * tanh_z;
            Real sech2 = 1.0 - tanh_z * tanh_z;

            EvalResult ev = compute_engine(rho);

            Real f = ev.value - market_el_val;

            // Chain Rule to Z
            Real drho_dz = 2.0 * tanh_z * sech2;
            Real d2rho_dz2 = 2.0 * sech2 * (1.0 - 3.0 * rho);

            Real f_z = f;
            Real f_prime_z = ev.grad * drho_dz;
            Real f_double_prime_z = ev.hess * (drho_dz * drho_dz) + ev.grad * d2rho_dz2;

            if (std::abs(f_z) < 1e-9) return rho;
            // Gradient vanishing check
            if (std::abs(f_prime_z) < 1e-12) return rho;

            // Halley Step
            Real num = 2.0 * f_z * f_prime_z;
            Real den = 2.0 * f_prime_z * f_prime_z - f_z * f_double_prime_z;
            
            Real delta = (std::abs(den) < 1e-10) ? (f_z / f_prime_z) : (num / den);
            
            z -= 0.8 * delta; // Damping 0.8
            
            if (z < 1e-3) z = 1e-3;
            if (z > 4.0) z = 4.0;
        }

        return std::tanh(z) * std::tanh(z);
    }
};

int main() {
    std::cout << "--- FlashBaseCorr ---" << std::endl;
    Real pd = 0.049;
    Real recovery = 0.4;
    Real detachment = 0.03;
    FlashBaseCorrSolver solver(pd, recovery, detachment);
    // Assume market EL is relatively high (low correlation)
    double rho = solver.solve(0.60 * detachment); 
    std::cout << "Implied Rho: " << rho << std::endl;
    return 0;
}