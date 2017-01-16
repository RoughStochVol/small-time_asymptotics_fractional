#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------
# BLACK-SCHOLES FRAMEWORK

# ------------------------------------------------------------------------------
# IMPORTS

import numpy as np
from numpy import inf
from math import sqrt, log, e, pi, isnan
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad

# ------------------------------------------------------------------------------
# CLASS DEFINITIONS


class Pricer:
    """
    Implementation of the usual Black-Scholes Pricer.

    May be used to compute European call option prices and some Greeks.

    :param flag: either "c" for calls or "p" for puts
    :param spot_price: spot price of the underlying
    :param strike: strike price
    :param time_to_maturity: time to maturity in years
    :param vol: annualized volatility assumed constant until expiration
    :param risk_free_rate: risk-free interest rate
    :param method: computation method for CDF function: "f" for built-in
                   formula, "n" for numerical integration

    :type flag: string
    :type spot_price: float
    :type strike: float
    :type time_to_maturity: float
    :type vol: float
    :type risk_free_rate: float
    :type method: string
    """

    def __init__(self, flag, spot_price, strike, time_to_maturity, vol,
                 risk_free_rate=0, method='f'):

        self.flag = flag
        self.S = spot_price
        self.K = strike
        self.T = time_to_maturity
        self.r = risk_free_rate
        self.vol = vol
        self.method = method

        self.d1 = 1/(self.vol * sqrt(self.T)) * \
            (log(self.S) - log(self.K) + (self.r + 1/2 * self.vol**2) *
             self.T)

        self.d2 = self.d1 - self.vol * sqrt(self.T)

        # If numerical method chosen, compute all quantiles numerically.
        if method == 'n':

            self.num_cdf_d1 = quad(norm.pdf, -inf, self.d1)[0]
            self.num_cdf_d2 = quad(norm.pdf, -inf, self.d2)[0]
            self.num_cdf_minus_d1 = quad(norm.pdf, -inf, -self.d1)[0]
            self.num_cdf_minus_d2 = quad(norm.pdf, -inf, -self.d2)[0]

    def get_price(self):
        """
        Computes the Black-Scholes price for the specified option.

        :return: Black-Scholes price
        :rtype: float
        """

        if self.flag == 'c':

            if self.method == "f":

                price = self.S * norm.cdf(self.d1) - self.K * \
                        e**(-self.r * self.T) * norm.cdf(self.d2)

            elif self.method == 'n':

                price = self.S * self.num_cdf_d1 - self.K * \
                        e**(-self.r * self.T) * self.num_cdf_d2

        elif self.flag == 'p':

            if self.method == 'f':

                price = self.K * e**(-self.r * self.T) * \
                        norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

            elif self.method == 'n':

                price = self.K * e**(-self.r * self.T) * \
                        self.num_cdf_minus_d2 - self.S * self.num_cdf_minus_d1

        return price

    def get_price_via_GBM(self, number_paths):
        """
        Computes the Black-Scholes price for the specified option via
        simulation of a Geometric Brownian motion.

        :param number_paths: number of sample paths
        :type number_paths: int

        :return: Black-Scholes price
        :rtype: float

        :return: standard deviation of estimator
        :rtype: float
        """

        W = np.random.randn(number_paths, 1)

        S = self.S * np.exp(-0.5 * self.vol**2 * self.T +
                            self.vol * sqrt(self.T) * W)

        if self.flag == 'c':

            payoffs = np.maximum(S - self.K, 0)

            price = np.average(payoffs)

            std_price = np.std(payoffs)/sqrt(number_paths)

        elif self.flag == 'p':

            payoffs = np.maximum(self.K - S, 0)

            price = np.average(payoffs)

            std_price = np.std(payoffs)/sqrt(number_paths)

        return price, std_price

    def get_delta(self):
        """
        Computes the Black-Scholes delta, the derivative of the option price
        with respect to the price of the underlying.

        :return: Black-Scholes delta
        :rtype: float
        """

        if self.flag == 'c':

            if self.method == 'f':

                return norm.cdf(self.d1)

            if self.method == 'n':

                return self.num_cdf_d1

        elif self.flag == 'p':

            if self.method == 'f':

                return norm.cdf(self.d1) - 1

            if self.method == 'n':

                return self.num_cdf_d1 - 1

    def get_vega(self):
        """
        Computes the Black-Scholes Vega, the derivative of the option price
        with respect to volatility.

        :return: Black-Scholes vega
        :rtype: float
        """

        vega = self.S * norm.pdf(self.d1) * sqrt(self.T)

        return vega


class ImpliedVol:
    """
    Computes the Black-Scholes implied volatility from a given market/model
    price. For the root-finding, we use Brent's method.

    :param flag: either "c" for calls or "p" for puts
    :param mkt_price: market/model option price to infer the implied vol from
    :param spot_price: spot price of the underlying
    :param strike: strike price
    :param time_to_maturity: time to maturity in years
    :param risk_free_rate: risk-free interest rate
    :param lower_bound: lower bound of implied volatility
    :param upper_bound: upper bound of implied volatility
    :param maxiter: maximum number of root finding iterations
    :param method: computation method for CDF function: "f" for built-in
                   formula, "n" for numerical integration

    :type flag: string
    :type mkt_price: float
    :type spot_price: float
    :type strike: float
    :type time_to_maturity: float
    :type risk_free_rate: float
    :type lower_bound: float
    :type upper_bound: float
    :type maxiter: integer
    :type method: string
    """

    def __init__(self, flag, mkt_price, spot_price, strike, time_to_maturity,
                 lower_bound, upper_bound, risk_free_rate=0, maxiter=1000,
                 method='f'):

        self.flag = flag
        self.mkt_price = mkt_price
        self.S = spot_price
        self.K = strike
        self.T = time_to_maturity
        self.r = risk_free_rate
        self.a = lower_bound
        self.b = upper_bound
        self.n = maxiter
        self.method = method

    def func(self, vol):
        """
        Objective function in root-finding problem.
        """

        p = Pricer(self.flag, self.S, self.K, self.T, vol, self.r, self.method)

        return p.get_price() - self.mkt_price

    def get(self):
        """
        Computes the Black-Scholes implied volatility.

        :return: Black-Scholes implied volatility
        :rtype: float
        """

        if self.mkt_price <= 0:

            implied_vol = float('NaN')

        else:

            implied_vol = brentq(self.func, self.a, self.b, maxiter=self.n)

        return implied_vol
