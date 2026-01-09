import time
import numpy as np
import scipy.special as sp
from scipy.integrate import quad
from scipy.optimize import newton

INV_SQRT_2PI = 0.3989422804014327

def norm_cdf(x):
    return sp.ndtr(x)

def norm_inv(p):
    return sp.ndtri(p)

def norm_pdf(x):
    return INV_SQRT_2PI * np.exp(-0.5 * x * x)

def bivariate_pdf(x, y, rho):
    denom = 1.0 - rho**2
    if denom < 1e-12: denom = 1e-12
    expo = (x**2 - 2*rho*x*y + y**2) / denom
    return (1.0 / (2.0 * np.pi * np.sqrt(denom))) * np.exp(-0.5 * expo)

def analytical_cdf_approx(h, k, rho):

    if rho > 0.9999: return norm_cdf(min(h, k))
    if abs(rho) < 1e-8: return norm_cdf(h) * norm_cdf(k)
    
    p1 = norm_cdf(h)
    p2 = norm_cdf(k)
    prod_pdf = norm_pdf(h) * norm_pdf(k)
    
    term1 = rho * prod_pdf
    term2 = (rho**2 / 2.0) * h * k * prod_pdf
    term3 = (rho**3 / 6.0) * (h*h - 1) * (k*k - 1) * prod_pdf
    
    return max(0.0, min(1.0, p1 * p2 + term1 + term2 + term3))

class TraditionalSolver:

    def __init__(self, pd, recovery, detachment):
        self.LGD = 1.0 - recovery
        self.K_prime = detachment / self.LGD
        self.alpha = norm_inv(pd)
        
    def _vasicek_cdf(self, x, rho):
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        sqrt_rho = np.sqrt(rho)
        thresh_x = norm_inv(x)
        numer = np.sqrt(1-rho) * thresh_x - self.alpha
        return norm_cdf(numer / sqrt_rho)

    def calculate_el(self, rho):
        integrand = lambda u: 1.0 - self._vasicek_cdf(u, rho)
        val, _ = quad(integrand, 0, self.K_prime, limit=50) 
        return self.LGD * val

    def solve(self, market_el):
        func = lambda r: self.calculate_el(r) - market_el
        try:
            rho, res = newton(func, 0.2, tol=1e-6, maxiter=50, full_output=True)
            return rho, res.iterations
        except:
            return np.nan, 0

class FlashCorrSolver:

    def __init__(self, pd, recovery, detachment):
        self.LGD = 1.0 - recovery
        self.K = detachment
        self.K_prime = detachment / self.LGD
        self.alpha = norm_inv(pd)
        self.beta = norm_inv(self.K_prime)

    def solve(self, market_el):
        z = 0.7 
        iters = 0
        
        for i in range(10):
            iters += 1
            tanh_z = np.tanh(z)
            rho = tanh_z**2
            rho = max(1e-5, min(1.0-1e-5, rho)) 
            
            sqrt_rho = np.sqrt(rho)
            r_copula = -sqrt_rho
            d2 = (self.beta * np.sqrt(1-rho) - self.alpha) / sqrt_rho
            
            phi2 = analytical_cdf_approx(self.alpha, d2, r_copula)
            term_b = self.K * (1.0 - norm_cdf(d2))
            el_val = self.LGD * phi2 + term_b
            
            f = el_val - market_el
            if abs(f) < 1e-9: break

            pdf2 = bivariate_pdf(self.alpha, d2, r_copula)
            d_rhocop_drho = -0.5 / sqrt_rho
            grad_rho = self.LGD * pdf2 * d_rhocop_drho

            h = 1e-5
            rho_up = rho + h
            sqrt_rho_up = np.sqrt(rho_up)
            d2_up = (self.beta * np.sqrt(1-rho_up) - self.alpha) / sqrt_rho_up
            pdf2_up = bivariate_pdf(self.alpha, d2_up, -sqrt_rho_up)
            grad_up = self.LGD * pdf2_up * (-0.5 / sqrt_rho_up)
            hess_rho = (grad_up - grad_rho) / h
            
            sech2 = 1.0 - tanh_z**2
            drho_dz = 2 * tanh_z * sech2
            d2rho_dz2 = 2 * sech2 * (1.0 - 3.0 * rho)
            
            f_z = f
            f_prime_z = grad_rho * drho_dz
            f_double_prime_z = hess_rho * drho_dz**2 + grad_rho * d2rho_dz2
            
            if abs(f_prime_z) < 1e-12: break
            num = 2 * f_z * f_prime_z
            den = 2 * f_prime_z**2 - f_z * f_double_prime_z
            
            if abs(den) < 1e-10: delta = f_z / f_prime_z # fallback
            else: delta = num / den
                
            z -= delta
            
        return tanh_z**2, iters


if __name__ == "__main__":
    pd = 0.05
    R = 0.4
    detachment = 0.03
    market_el = 0.015

    print(f"--- Benchmark: Traditional vs FlashCorr ---")
    
    trad_solver = TraditionalSolver(pd, R, detachment)
    flash_solver = FlashCorrSolver(pd, R, detachment)

    N_RUNS = 100 

    # Test Traditional
    t0 = time.time()
    total_iter_t = 0
    for _ in range(N_RUNS):
        _, it = trad_solver.solve(market_el)
        total_iter_t += it
    t1 = time.time()
    avg_time_trad = (t1 - t0) / N_RUNS

    # Test FlashCorr
    t0 = time.time()
    total_iter_f = 0
    for _ in range(N_RUNS):
        _, it = flash_solver.solve(market_el)
        total_iter_f += it
    t1 = time.time()
    avg_time_flash = (t1 - t0) / N_RUNS

    print(f"\n{'Method':<15} | {'Avg Time (ms)':<15} | {'Iterations':<10} | {'Speedup'}")
    print("-" * 60)
    print(f"{'Traditional':<15} | {avg_time_trad*1000:<15.4f} | {total_iter_t/N_RUNS:<10.1f} | 1.0x")
    print(f"{'FlashCorr':<15} | {avg_time_flash*1000:<15.4f} | {total_iter_f/N_RUNS:<10.1f} | {avg_time_trad/avg_time_flash:.1f}x")