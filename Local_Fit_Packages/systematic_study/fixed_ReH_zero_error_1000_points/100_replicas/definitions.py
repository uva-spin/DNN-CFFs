"""
Definitions module combining classes from BHDVCS_tf, dvcs_code, and km15.
This consolidates all major classes needed for the physics calculations.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import math
from scipy.integrate import quad
from math import pi, sqrt, pow

# ============================================================================
# From BHDVCS_tf_modified.py
# ============================================================================

class TotalFLayer(tf.keras.layers.Layer):
    """TensorFlow layer that computes total F using BHDVCStf."""
    def __init__(self, **kwargs):
        super(TotalFLayer, self).__init__(**kwargs)
        self.f = BHDVCStf()

    def call(self, inputs):
        return self.f.curve_fit(inputs[:, 0:5], inputs[:, 5:9])  # QQ, x, t, phi, k, cff1, cff2, cff3, cff4


class BHDVCStf(object):
    """Bethe-Heitler Deeply Virtual Compton Scattering TensorFlow implementation."""

    def __init__(self):
        self.ALP_INV = tf.constant(137.0359998)  # 1 / Electromagnetic Fine Structure Constant
        self.PI = tf.constant(3.1415926535)
        self.RAD = tf.constant(self.PI / 180.)
        self.M = tf.constant(0.938272)  # Mass of the proton in GeV
        self.GeV2nb = tf.constant(.389379 * 1000000)  # Conversion from GeV to NanoBar
        self.M2 = tf.constant(0.938272 * 0.938272)  # Mass of the proton squared in GeV

    @tf.function
    def SetKinematics(self, QQ, x, t, k):
        ee = 4. * self.M2 * x * x / QQ  # epsilon squared
        y = tf.sqrt(QQ) / (tf.sqrt(ee) * k)  # lepton energy fraction
        xi = x * (1. + t / 2. / QQ) / (2. - x + x * t / QQ)  # Generalized Bjorken variable
        Gamma = x * y * y / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / 8. / QQ / QQ / tf.sqrt(1. + ee)  # factor in front of the cross section, eq. (22)
        tmin = - QQ * (2. * (1. - x) * (1. - tf.sqrt(1. + ee)) + ee) / (4. * x * (1. - x) + ee)  # eq. (31)
        Ktilde_10 = tf.sqrt(tmin - t) * tf.sqrt((1. - x) * tf.sqrt(1. + ee) + ((t - tmin) * (ee + 4. * x * (1. - x)) / 4. / QQ)) * tf.sqrt(1. - y - y * y * ee / 4.) / tf.sqrt(1. - y + y * y * ee / 4.)  # K tilde from 2010 paper
        K = tf.sqrt(1. - y + y * y * ee / 4.) * Ktilde_10 / tf.sqrt(QQ)
        return ee, y, xi, Gamma, tmin, Ktilde_10, K

    @tf.function
    def BHLeptonPropagators(self, phi, QQ, x, t, ee, y, K):
        # KD 4-vector product (phi-dependent)
        KD = - QQ / (2. * y * (1. + ee)) * (1. + 2. * K * tf.cos(self.PI - (phi * self.RAD)) - t / QQ * (1. - x * (2. - y) + y * ee / 2.) + y * ee / 2.)  # eq. (29)

        # lepton BH propagators P1 and P2 (contaminating phi-dependence)
        P1 = 1. + 2. * KD / QQ
        P2 = t / QQ - 2. * KD / QQ
        return P1, P2

    @tf.function
    def BHUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K):
        # BH unpolarized Fourier harmonics eqs. (35 - 37)
        phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K = [tf.cast(i, np.float32) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K]]
        c0_BH = 8. * K * K * ((2. + 3. * ee) * (QQ / t) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * x * x * (F1 + F2) * (F1 + F2)) + (2. - y) * (2. - y) * ((2. + ee) * (
                    (4. * x * x * self.M2 / t) * (1. + t / QQ) * (
                        1. + t / QQ)
                        + 4. * (1. - x) * (1. + x * t / QQ)) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 4. * x * x * (x + (1. - x + ee / 2.) * (1. - t / QQ) * (1. - t / QQ) - x * (1. - 2. * x) * t * t / (QQ * QQ)) * (F1 + F2) * (F1 + F2)) + 8. * (
                                 1. + ee) * (1. - y - ee * y * y / 4.) * (
                                 2. * ee * (1. - t / (4. * self.M2)) * (
                                     F1 * F1 - F2 * F2 * t / (4. * self.M2)) - x * x * (
                                             1. - t / QQ) * (1. - t / QQ) * (F1 + F2) * (F1 + F2))

        c1_BH = 8. * K * (2. - y) * (
                    (4. * x * x * self.M2 / t - 2. * x - ee) * (
                        F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * x * x * (
                                1. - (1. - 2. * x) * t / QQ) * (F1 + F2) * (F1 + F2))

        c2_BH = 8. * x * x * K * K * (
                    (4. * self.M2 / t) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * (F1 + F2) * (
                        F1 + F2))

        # BH squared amplitude eq (25) divided by e^6
        Amp2_BH = 1. / (x * x * y * y * (1. + ee) * (
                    1. + ee) * t * P1 * P2) * (c0_BH + c1_BH * tf.cos(
            self.PI - (phi * self.RAD)) + c2_BH * tf.cos(2. * (self.PI - (phi * self.RAD))))

        Amp2_BH = self.GeV2nb * Amp2_BH  # convertion to nb

        return Gamma * Amp2_BH

    @tf.function
    def IUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, twist, tmin, xi, Ktilde_10):
        phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10 = [tf.cast(i, np.float32) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10]]
        # Get BH propagators and set the kinematics
        self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)

        # Get A_UU_I, B_UU_I and C_UU_I interference coefficients
        A_U_I, B_U_I, C_U_I = self.ABC_UU_I_10(phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10)

        # BH-DVCS interference squared amplitude
        I_10 = 1. / (x * y * y * y * t * P1 * P2) * (
                    A_U_I * (F1 * ReH - t / 4. / self.M2 * F2 * ReE) + B_U_I * (F1 + F2) * (
                        ReH + ReE) + C_U_I * (F1 + F2) * ReHtilde)

        I_10 = self.GeV2nb * I_10  # convertion to nb

        return Gamma * I_10

    @tf.function
    def ABC_UU_I_10(self, phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10):  # Get A_UU_I, B_UU_I and C_UU_I interference coefficients BKM10

        if twist == "t2":
            f = 0  # F_eff = 0 ( pure twist 2)
        if twist == "t3":
            f = - 2. * xi / (1. + xi)
        if twist == "t3ww":
            f = 2. / (1. + xi)

        # Interference coefficients  (BKM10 Appendix A.1)
        # n = 0 -----------------------------------------
        # helicity - conserving (F)
        C_110 = - 4. * (2. - y) * (1. + tf.sqrt(1 + ee)) / tf.pow((1. + ee), 2) * (
                    Ktilde_10 * Ktilde_10 * (2. - y) * (2. - y) / QQ / tf.sqrt(1 + ee)
                    + t / QQ * (1. - y - ee / 4. * y * y) * (2. - x) * (1. + (
                        2. * x * (2. - x + (tf.sqrt(
                    1. + ee) - 1.) / 2. + ee / 2. / x) * t / QQ + ee) / (2. - x) / (
                                                                                                                       1. + tf.sqrt(
                                                                                                                   1. + ee))))
        C_110_V = 8. * (2. - y) / tf.pow((1. + ee), 2) * x * t / QQ * (
                    (2. - y) * (2. - y) / tf.sqrt(1. + ee) * Ktilde_10 * Ktilde_10 / QQ
                    + (1. - y - ee / 4. * y * y) * (1. + tf.sqrt(1. + ee)) / 2. * (
                                1. + t / QQ) * (1. + (tf.sqrt(1. + ee) - 1. + 2. * x) / (
                        1. + tf.sqrt(1. + ee)) * t / QQ))
        C_110_A = 8. * (2. - y) / tf.pow((1. + ee), 2) * t / QQ * (
                    (2. - y) * (2. - y) / tf.sqrt(
                1. + ee) * Ktilde_10 * Ktilde_10 / QQ * (
                                1. + tf.sqrt(1. + ee) - 2. * x) / 2.
                    + (1. - y - ee / 4. * y * y) * ((1. + tf.sqrt(1. + ee)) / 2. * (
                        1. + tf.sqrt(1. + ee) - x + (
                            tf.sqrt(1. + ee) - 1. + x * (3. + tf.sqrt(1. + ee) - 2. * x) / (
                                1. + tf.sqrt(1. + ee)))
                        * t / QQ) - 2. * Ktilde_10 * Ktilde_10 / QQ))
        # helicity - changing (F_eff)
        C_010 = 12. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 ee + (2. - 6. * x - ee) / 3. * t / QQ)
        C_010_V = 24. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * x * t / QQ * (
                                   1. - (1. - 2. * x) * t / QQ)
        C_010_A = 4. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * t / QQ * (
                                   8. - 6. * x + 5. * ee) * (
                                   1. - t / QQ * ((2. - 12 * x * (1. - x) - ee)
                                                            / (8. - 6. * x + 5. * ee)))
        # n = 1 -----------------------------------------
        # helicity - conserving (F)
        C_111 = -16. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * ((1. + (1. - x) * (tf.sqrt(
            1 + ee) - 1.) / 2. / x + ee / 4. / x) * x * t / QQ - 3. * ee / 4.) - 4. * K * (
                                 2. - 2. * y + y * y + ee / 2. * y * y) * (
                                 1. + tf.sqrt(1 + ee) - ee) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 1. - (1. - 3. * x) * t / QQ + (
                                     1. - tf.sqrt(1 + ee) + 3. * ee) / (
                                             1. + tf.sqrt(1 + ee) - ee) * x * t / QQ)
        C_111_V = 16. * K / tf.pow(tf.sqrt(1. + ee), 5) * x * t / QQ * (
                    (2. - y) * (2. - y) * (1. - (1. - 2. * x) * t / QQ) + (
                        1. - y - ee / 4. * y * y)
                    * (1. + tf.sqrt(1. + ee) - 2. * x) / 2. * (t - tmin) / QQ)
        C_111_A = -16. * K / tf.pow((1. + ee), 2) * t / QQ * (
                    (1. - y - ee / 4. * y * y) * (1. - (1. - 2. * x) * t / QQ + (
                        4. * x * (1. - x) + ee) / 4. / tf.sqrt(1. + ee) * (
                                                                                  t - tmin) / QQ)
                    - tf.pow((2. - y), 2) * (
                                1. - x / 2. + (1. + tf.sqrt(1. + ee) - 2. * x) / 4. * (
                                    1. - t / QQ) + (4. * x * (1. - x) + ee) / 2. / tf.sqrt(
                            1. + ee) * (t - tmin) / QQ))
        # helicity - changing (F_eff)
        C_011 = 8. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            (1. + ee), 2) * (tf.pow((2. - y), 2) * (t - tmin) / QQ * (
                    1. - x + ((1. - x) * x + ee / 4.) / tf.sqrt(1. + ee) * (
                        t - tmin) / QQ)
                                  + (1. - y - ee / 4. * y * y) / tf.sqrt(1 + ee) * (
                                              1. - (1. - 2. * x) * t / QQ) * (
                                              ee - 2. * (1. + ee / 2. / x) * x * t / QQ))
        C_011_V = 16. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * x * t / QQ * (
                                   tf.pow(Ktilde_10 * (2. - y), 2) / QQ + tf.pow(
                              (1. - (1. - 2. * x) * t / QQ), 2) * (
                                               1. - y - ee / 4. * y * y))
        C_011_A = 8. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * t / QQ * (
                                   tf.pow(Ktilde_10 * (2. - y), 2) * (1. - 2. * x) / QQ + (
                                       1. - (1. - 2. * x) * t / QQ)
                                   * (1. - y - ee / 4. * y * y) * (
                                               4. - 2. * x + 3. * ee + t / QQ * (
                                                   4. * x * (1. - x) + ee)))
        # n = 2 -----------------------------------------
        # helicity - conserving (F)
        C_112 = 8. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                     2) * (
                                 2. * ee / tf.sqrt(1. + ee) / (1. + tf.sqrt(1. + ee)) * tf.pow(
                             Ktilde_10, 2) / QQ + x * t * (
                                             t - tmin) / QQ / QQ * (1. - x - (
                                     tf.sqrt(1. + ee) - 1.) / 2. + ee / 2. / x))
        C_112_V = 8. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                       2) * x * t / QQ * (
                                   4. * tf.pow(Ktilde_10, 2) / tf.sqrt(1. + ee) / QQ + (
                                       1. + tf.sqrt(1. + ee) - 2. * x) / 2. * (1. + t / QQ) * (
                                               t - tmin) / QQ)
        C_112_A = 4. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                       2) * t / QQ * (
                                   4. * (1. - 2. * x) * tf.pow(Ktilde_10, 2) / tf.sqrt(
                               1. + ee) / QQ - (3. - tf.sqrt(
                               1. + ee) - 2. * x + ee / x) * x * (
                                               t - tmin) / QQ)
        # helicity - changing (F_eff)
        C_012 = -8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 1. + ee / 2.) * (
                                 1. + (1. + ee / 2. / x) / (1. + ee / 2.) * x * t / QQ)
        C_012_V = 8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * x * t / QQ * (
                                   1. - (1. - 2. * x) * t / QQ)
        C_012_A = 8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow((1. + ee), 2) * t / QQ * (
                                   1. - x + (t - tmin) / 2. / QQ * (
                                       4. * x * (1. - x) + ee) / tf.sqrt(1. + ee))
        # n = 3 -----------------------------------------
        # helicity - conserving (F)
        C_113 = -8. * K * (1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                                               5) * (
                                 tf.sqrt(1. + ee) - 1.) * (
                                 (1. - x) * t / QQ + (tf.sqrt(1. + ee) - 1.) / 2. * (
                                     1. + t / QQ))
        C_113_V = -8. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * x * t / QQ * (tf.sqrt(1. + ee) - 1. + (
                    1. + tf.sqrt(1. + ee) - 2. * x) * t / QQ)
        C_113_A = 16. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * t * (t - tmin) / QQ / QQ * (
                                   x * (1. - x) + ee / 4.)

        # A_U_I, B_U_I and C_U_I
        A_U_I = C_110 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
            QQ) * f * C_010 + (C_111 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
            QQ) * f * C_011) * tf.cos(self.PI - (phi * self.RAD)) + (
                                 C_112 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                             QQ) * f * C_012) * tf.cos(
            2. * (self.PI - (phi * self.RAD))) + C_113 * tf.cos(3. * (self.PI - (phi * self.RAD)))
        B_U_I = xi / (1. + t / 2. / QQ) * (
                    C_110_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * C_010_V + (
                                C_111_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * C_011_V) * tf.cos(self.PI - (phi * self.RAD)) + (
                                C_112_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * C_012_V) * tf.cos(
                2. * (self.PI - (phi * self.RAD))) + C_113_V * tf.cos(3. * (self.PI - (phi * self.RAD))))
        C_U_I = xi / (1. + t / 2. / QQ) * (
                    C_110 + C_110_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * (C_010 + C_010_A) + (
                                C_111 + C_111_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * (C_011 + C_011_A)) * tf.cos(self.PI - (phi * self.RAD)) + (
                                C_112 + C_112_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * (C_012 + C_012_A)) * tf.cos(
                2. * (self.PI - (phi * self.RAD))) + (C_113 + C_113_A) * tf.cos(
                3. * (self.PI - (phi * self.RAD))))

        return A_U_I, B_U_I, C_U_I

    @tf.function
    def curve_fit(self, kins, cffs):
        calc = F1F2()
        QQ, x, t, phi, k = tf.split(kins, num_or_size_splits=5, axis=1)
        F1, F2 = calc.f1_f21(t)  # calculating F1 and F2 using passed data as opposed to passing in F1 and F2
        ReH, ReE, ReHtilde, c0fit = tf.split(cffs, num_or_size_splits=4, axis=1)  # output of network
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.SetKinematics(QQ, x, t, k)
        P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return f_pred


class DvcsData(object):
    """Data handler for DVCS data."""
    def __init__(self, df):
        self.df = df
        self.X = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs']]
        self.XnoCFF = df.loc[:, ['phi_x', 'k']]  # removed redundant data
        self.y = df.loc[:, 'F']
        self.Kinematics = df.loc[:, ['QQ', 'x_b', 't']]  # Removed k from kinematics
        self.erry = df.loc[:, 'errF']

    def getSet(self, setNum, itemsInSet=36):
        pd.options.mode.chained_assignment = None
        subX = self.X.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1, :]
        subX['F'] = self.y.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1]
        subX['errF'] = self.erry.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1]
        pd.options.mode.chained_assignment = 'warn'
        return DvcsData(subX)

    def __len__(self):
        return len(self.X)

    def sampleY(self):
        return np.random.normal(self.y, self.erry)

    def sampleWeights(self):
        return 1/self.erry

    def getAllKins(self, itemsInSets=36):
        return self.Kinematics.iloc[np.array(range(len(self.df)//itemsInSets))*itemsInSets, :]


class F1F2:
    """Form factors F1 and F2 calculation."""
    def __init__(self):
        self.GM0 = 2.792847337

    def ffGE(self, t):
        GE = 1.0 / (1.0 + (-t / 0.710649)) / (1.0 + (-t / 0.710649))
        return GE

    def ffGM(self, t):
        shape = self.ffGE(t)
        return self.GM0 * shape

    def ffF2(self, t):
        f2 = (self.ffGM(t) - self.ffGE(t)) / (1.0 - t / (4.0 * 0.938272 * 0.938272))
        return f2

    def ffF1(self, t):
        f1 = self.ffGM(t) - self.ffF2(t)
        return f1

    def f1_f21(self, t):
        return self.ffF1(t), self.ffF2(t)


class F_calc:
    """F calculation wrapper."""
    def __init__(self):
        self.module = BHDVCStf()

    def fn_1(self, kins, cffs):
        phi, QQ, x, t, k, F1, F2 = kins
        ReH, ReE, ReHtilde, c0fit = cffs
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.module.SetKinematics(QQ, x, t, k)
        P1, P2 = self.module.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.module.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.module.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return tf.get_static_value(f_pred)


# ============================================================================
# From dvcs_code.py (BKM10 implementation)
# ============================================================================

import keras


class BKM10(keras.layers.Layer):
    """BKM10 formalism implementation for DVCS calculations."""

    def __init__(self, k, Q2, x, t, helc=0, twist=2):
        '''
        kins(sequence of float, shape(5,)): Kinematic variables[k, Q², x_b, t, φ] where
        - k  : incoming lepton energy in the target rest frame(GeV)
        - Q² : virtuality of the photon(GeV²)
        - x  : Bjorken x (dimensionless)
        - t  : negative squared momentum transfer(GeV²)
        '''

        super().__init__()

        self.PI = tf.constant(3.141592653589793, dtype=tf.float32)
        self.ALPHA = tf.constant(1/137.0359998, dtype=tf.float32)
        self.M2 = tf.constant(0.93827208816*0.93827208816, dtype=tf.float32)
        self.GeV_to_nb = tf.constant(0.389379e6, dtype=tf.float32)

        self.k = k
        self.Q2 = Q2
        self.x = x
        self.t = t
        self.helc = helc

        # BKM 2010 : Eq (1.2)
        self.e2 = 4.*x*x*self.M2/Q2
        e2 = self.e2

        # Definition of lepton energy loss. Maybe from DIS theory?
        self.y = tf.sqrt(Q2/e2)/k
        y = self.y

        # BKM 2002 : Eq (4)
        self.xi = x * (1.+0.5*t/Q2)/(2.-x+x*t/Q2)

        # BKM 2002 : Eq (31)
        self.t_min = -Q2*(2.*(1.-x)*(1.-tf.sqrt(1+e2))+e2)/(4.*x*(1.-x)+e2)
        t_min = self.t_min

        # BKM 2010 : EQ (2.5) and BKM 2002 : EQ (30)?
        self.K_tilde = tf.sqrt((t_min-t)*((1.-x)*tf.sqrt(1.+e2) + (t-t_min)*(e2+4.*x*(1.-x))/(4.*Q2)) *
                               (1.-y-y*y*e2/4.)/(1.-y+y*y*e2/4.))
        K_tilde = self.K_tilde

        # BKM 2010 : Below Eq (2.21)
        self.K = tf.sqrt((1.-y+e2*y*y*0.25)/Q2)*K_tilde

        self.F1, self.F2 = self.form_factors(t)

        # BKM 2010 : prefactor in Eq (1.1)
        self.dsig_prefactor = self.ALPHA*self.ALPHA*self.ALPHA * \
            x*y*y/(8.*self.PI*Q2*Q2*tf.sqrt(1.+e2))
        
        # factor = 0 is twist 2
        # for a slight inclusion of twist 3, put factor as the first term of BKM 2002 : Eq (52)
        if twist==2:
            self.factor = 0
        else:
            self.factor = -2.*self.xi/(1.+self.xi)

    def diff_cross(self, phi, cffs):
        """
        Computes the differential cross section for the BKM formalism in twist-2 for now,
        based on Eqn (1.1) BKM 2010 paper.

        Parameters:
            φ : azimuthal angle between leptonic and hadronic planes (radians)
            cffs ((sequence of float, shape (8,))) : 
                Compton form factors [Re(H), Re(H̃), Re(E), Re(Ẽ), Im(H), Im(H̃), Im(E), Im(Ẽ)].

        Returns:
            d_sigma : float
                Differential cross section d⁴σ/(dQ² dt dx_b dΦ), in nb.
        """

        # BKM 2002 : Eq (29)
        kDelta = -(self.Q2/(2.*self.y*(1.+self.e2))) * (1. + 2.*self.K*tf.cos(phi) -
                                                        (self.t/self.Q2)*(1.-self.x*(2.-self.y)+0.5*self.y*self.e2) + 0.5*self.y*self.e2)

        # BKM 2002 : Eq (28)
        P1 = 1. + (2.*kDelta)/self.Q2
        P2 = (-2.*kDelta+self.t)/self.Q2

        ReH, ReHt, ReE, ReEt, ImH, ImHt, ImE, ImEt = tf.unstack(cffs, axis=1)

        T_BH_2 = self.compute_BH_amplitude(phi, P1, P2)

        I = self.compute_Interference(phi, P1, P2, ReH, ImH, ReHt, ImHt, ReE, ImE)

        T_DVCS_2 = self.compute_DVCS_amplitude(phi, ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt)

        d_sigma = T_BH_2 + T_DVCS_2 + I
        
        return d_sigma, T_BH_2, T_DVCS_2, I

    def compute_BH_amplitude(self, phi, P1, P2):
        """Computes the Bethe-Heitler amplitude squared term in nb using inputs from compute_kinematics"""
        c0_BH, c1_BH, c2_BH, s1_BH = self.BH_UP_coeffs()

        # BKM 2002 : Eq (25)
        TBH2 = 1./(self.x*self.x*self.y*self.y*(1.+self.e2)*(1.+self.e2)*self.t*P1*P2) * \
            (c0_BH + c1_BH*tf.cos(phi) + c2_BH *
             tf.cos(2.*phi) + s1_BH*tf.sin(phi))

        TBH2 *= self.GeV_to_nb*self.dsig_prefactor

        return TBH2

    def compute_DVCS_amplitude(self, phi, ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt):
        """Computes the DVCS amplitude squared term in nb using inputs from compute_kinematics"""
        c0_DVCS, c1_DVCS, s1_DVCS = self.DVCS_UP_coeffs(
            ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt)

        # BKM 2010 : Eq (2.17)
        TDVCS2 = 1./(self.y*self.y*self.Q2) * (c0_DVCS + c1_DVCS *
                                               tf.cos(phi) + s1_DVCS*tf.sin(phi))

        TDVCS2 *= self.GeV_to_nb*self.dsig_prefactor

        return TDVCS2

    def compute_Interference(self, phi, P1, P2, ReH, ImH, ReHt, ImHt, ReE, ImE):
        """Computes the Interference term amplitude squared term in nb using inputs from compute_kinematics"""
        c0_I, c1_I, c2_I, c3_I, s1_I, s2_I = self.I_UP_coeffs(ReH, ImH, ReHt, ImHt, ReE, ImE)

        # BKM 2010 : Eq (2.34)
        I = 1./(self.x*self.y*self.y*self.y*self.t*P1*P2) * \
            (c0_I + c1_I*tf.cos(phi) + c2_I*tf.cos(2.*phi) +
             c3_I*tf.cos(3.*phi) + s1_I*tf.sin(phi) + s2_I*tf.sin(2.*phi))

        I *= self.GeV_to_nb*self.dsig_prefactor

        return I

    def form_factors(self, t):
        """
        Computes the Dirac (F1) and Pauli (F2) proton form factors using the dipole approximation for Sachs form factors.

        Parameters:
            t (float): Momentum transfer squared (GeV^2)

        Returns:
            tuple: (F1, F2)
                F1 (float): Dirac form factor
                F2 (float): Pauli form factor
        """

        GM0 = 2.792847337  # Proton magnetic moment

        # Kelly's parametrization fit Parameters
        a1_GEp = -0.24
        b1_GEp = 10.98
        b2_GEp = 12.82
        b3_GEp = 21.97
        a1_GMp = 0.12
        b1_GMp = 10.97
        b2_GMp = 18.86
        b3_GMp = 6.55

        tau = - 0.25*t/self.M2
        GEp = (1.+a1_GEp*tau)/(1.+b1_GEp*tau+b2_GEp*tau*tau+b3_GEp*tau*tau*tau)
        GMp = GM0*(1.+a1_GMp*tau)/(1.+b1_GMp*tau +
                                   b2_GMp*tau*tau+b3_GMp*tau*tau*tau)

        F2 = (GMp-GEp)/(1.+tau)  # Pauli FF
        F1 = GMp-F2  # Dirac FF

        return F1, F2

    def BH_UP_coeffs(self):
        """Computes the Fourier coefficients (c0, c1, c2, s1) of the unpolarized target Bethe-Heitler squared amplitude."""
        Q2, t, x, e2, y, K, F1, F2 = self.Q2, self.t, self.x, self.e2, self.y, self.K, self.F1, self.F2

        A = (F1*F1-(t/(4.*self.M2))*F2*F2)
        B = (F1+F2)*(F1+F2)

        c0 = (
            8.*K*K*((2.+3.*e2)*(Q2/t)*A + 2.*x*x*B) +
            (2.-y)*(2.-y) * (
                (2.+e2)*((4.*x*x*self.M2/t)*(1.+t/Q2)*(1.+t/Q2) + 4.*(1.-x)*(1.+x*t/Q2))*A +
                4.*x*x*(x+(1.-x+0.5*e2)*(1.-t/Q2) *
                        (1.-t/Q2)-x*(1.-2.*x)*(t*t/(Q2*Q2)))*B
            ) +
            8.*(1.+e2)*(1.-y-e2*y*y/4.) *
            (2.*e2*(1.-t/(4.*self.M2))*A-x*x*(1.-t/Q2)*(1.-t/Q2)*B)
        )

        c1 = 8.*K*(2.-y)*((4.*x*x*self.M2/t-2.*x-e2)
                          * A + 2.*x*x*(1.-(1.-2.*x)*t/Q2)*B)

        c2 = 8.*x*x*K*K*((4.*self.M2/t)*A + 2.*B)

        s1 = 0.

        return c0, c1, c2, s1

    def DVCS_UP_coeffs(self, ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt):
        """Computes Fourier coefficients for the DVCS term in the unpolarized Bethe-Heitler + DVCS cross section."""
        Q2, t, x, e2, y, K = self.Q2, self.t, self.x, self.e2, self.y, self.K

        def CCDVCS_UP_of_F_Fstar(ReH_, ImH_, ReHt_, ImHt_, ReE_, ImE_, ReEt_, ImEt_,
                                 ReHs_, ImHs_, ReHts_, ImHts_, ReEs_, ImEs_, ReEts_, ImEts_):
            # BKM 2010 : Eq (2.22)
            A = ((2.-x)*Q2+x*t) * ((2.-x)*Q2+x*t)
            Qxt = Q2+x*t
            Qt = Q2+t

            Re_HHstar = ReH_*ReHs_ + ImH_*ImHs_
            Im_HHstar = ImH_*ReHs_ - ReH_*ImHs_
            Re_EEstar = ReE_*ReEs_ + ImE_*ImEs_
            Im_EEstar = ImE_*ReEs_ - ReE_*ImEs_

            Re_HtHtstar = ReHt_*ReHts_ + ImHt_*ImHts_
            Im_HtHtstar = ImHt_*ReHts_ - ReHt_*ImHts_
            Re_EtEtstar = ReEt_*ReEts_ + ImEt_*ImEts_
            Im_EtEtstar = ImEt_*ReEts_ - ReEt_*ImEts_

            Re_HEstar = ReH_*ReEs_ + ImH_*ImEs_
            Im_HEstar = ImH_*ReEs_ - ReH_*ImEs_
            Re_EHstar = ReHs_*ReE_ + ImHs_*ImE_
            Im_EHstar = ReHs_*ImE_ - ImHs_*ReE_

            Re_HtEtstar = ReHt_*ReEts_ + ImHt_*ImEts_
            Im_HtEtstar = ImHt_*ReEts_ - ReHt_*ImEts_
            Re_EtHtstar = ReHts_*ReEt_ + ImHts_*ImEt_
            Im_EtHtstar = ReHts_*ImEt_ - ImHts_*ReEt_

            ReCC = (Q2*Qxt/A) *\
                (
                    4.*(1.-x)*Re_HHstar+4.*(1.-x+(2.*Q2+t)*0.25*e2/Qxt)*Re_HtHtstar -
                    x*x*Qt*Qt*(Re_HEstar+Re_EHstar)/(Q2*Qxt) - x*x*Q2*(Re_HtEtstar+Re_EtHtstar)/Qxt -
                    (x*x*Qt*Qt/(Q2*Qxt) + A*t/(Q2*Qxt*4*self.M2)) *
                Re_EEstar - (x*x*Q2*t/(Qxt*4*self.M2))*Re_EtEtstar
                )

            ImCC = (Q2*Qxt/A) *\
                (
                4.*(1.-x)*Im_HHstar+4.*(1.-x+(2.*Q2+t)*0.25*e2/Qxt)*Im_HtHtstar -
                x*x*Qt*Qt*(Im_HEstar+Im_EHstar)/(Q2*Qxt) - x*x*Q2*(Im_HtEtstar+Im_EtHtstar)/Qxt -
                (x*x*Qt*Qt/(Q2*Qxt) + A*t/(Q2*Qxt*4*self.M2)) *
                Im_EEstar - (x*x*Q2*t/(Qxt*4*self.M2))*Im_EtEtstar
            )

            return ReCC, ImCC

        # First term of BKM 2010 : Eq (2.18)
        Re_CC0A, Im_CC0A = CCDVCS_UP_of_F_Fstar(ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt,
                                                ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt)

        c0_A = 2.*(2.-2.*y+y*y+0.5*e2*y*y)*Re_CC0A/(1.+e2)

        ReH_eff = ReH * self.factor
        ImH_eff = ImH * self.factor
        ReHt_eff = ReHt * self.factor
        ImHt_eff = ImHt * self.factor
        ReE_eff = ReE * self.factor
        ImE_eff = ImE * self.factor
        ReEt_eff = ReE * self.factor
        ImEt_eff = ImE * self.factor

        # Second term of BKM 2010 : Eq (2.18); First term of BKM 2010 : Eq (2.19)
        Re_CC0B, Im_CC0B = CCDVCS_UP_of_F_Fstar(ReH_eff, ImH_eff, ReHt_eff, ImHt_eff, ReE_eff, ImE_eff, ReEt_eff, ImEt_eff,
                                                ReH_eff, ImH_eff, ReHt_eff, ImHt_eff, ReE_eff, ImE_eff, ReEt_eff, ImEt_eff)

        c0_B = 16.*K*K*Re_CC0B/((2.-x)*(2.-x)*(1.+e2))

        Re_CC1, Im_CC1 = CCDVCS_UP_of_F_Fstar(ReH_eff, ImH_eff, ReHt_eff, ImHt_eff, ReE_eff, ImE_eff, ReEt_eff, ImEt_eff,
                                              ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt)

        c1_A = 8.*K*(2.-y)*Re_CC1/((2.-x)*(1.+e2))

        s1_A = 8.*K*(-self.helc*y*tf.sqrt(1.+e2))*Im_CC1/((2.-x)*(1.+e2))

        c0 = c0_A + c0_B
        c1 = c1_A
        s1 = s1_A

        return c0, c1, s1

    def I_UP_coeffs(self, ReH, ImH, ReHt, ImHt, ReE, ImE):
        """Computes Fourier coefficients for the interference term."""
        def CCI_UP_of_F(ReH_, ImH_, ReHt_, ImHt_, ReE_, ImE_):
            A = (self.x/(2.-self.x+self.x*self.t/self.Q2))

            # BKM 2010 : Eq (2.28)
            ReCC_ = self.F1*ReH_-(0.25*self.t/self.M2)*self.F2*ReE_+A*(self.F1+self.F2)*ReHt_
            ImCC_ = self.F1*ImH_-(0.25*self.t/self.M2)*self.F2*ImE_+A*(self.F1+self.F2)*ImHt_

            # BKM 2010 : Eq (2.29)
            ReCC_V_ = A*(self.F1+self.F2)*(ReH_+ReE_)
            ImCC_V_ = A*(self.F1+self.F2)*(ImH_+ImE_)

            # BKM 2010 : Eq (2.30)
            ReCC_A_ = A*(self.F1+self.F2)*ReHt_
            ImCC_A_ = A*(self.F1+self.F2)*ImHt_

            return ReCC_, ImCC_, ReCC_V_, ImCC_V_, ReCC_A_, ImCC_A_

        ReCC, ImCC, ReCC_V, ImCC_V, ReCC_A, ImCC_A = CCI_UP_of_F(
            ReH, ImH, ReHt, ImHt, ReE, ImE)

        # First term of BKM 2010 : Eq (2.35)
        c0_A, c1_A, c2_A, c3_A, s1_A, s2_A = self.coeffs_plus_plus_UP(
            ReCC, ReCC_V, ReCC_A, ImCC, ImCC_V, ImCC_A)

        ReH_eff = ReH * self.factor
        ImH_eff = ImH * self.factor
        ReHt_eff = ReHt * self.factor
        ImHt_eff = ImHt * self.factor
        ReE_eff = ReE * self.factor
        ImE_eff = ImE * self.factor

        ReCC_eff, ImCC_eff, ReCC_V_eff, ImCC_V_eff, ReCC_A_eff, ImCC_A_eff = CCI_UP_of_F(
            ReH_eff, ImH_eff, ReHt_eff, ImHt_eff, ReE_eff, ImE_eff)

        # Second term of BKM 2010 : Eq (2.35)
        c0_B, c1_B, c2_B, c3_B, s1_B, s2_B = self.coeffs_zero_plus_UP(
            ReCC_eff, ReCC_V_eff, ReCC_A_eff, ImCC_eff, ImCC_V_eff, ImCC_A_eff)

        # Third term of BKM 2010 : Eq (2.35) - Skip for now
        c0_C, c1_C, c2_C, c3_C, s1_C, s2_C = 0., 0., 0., 0., 0., 0.

        c0 = c0_A + c0_B + c0_C
        c1 = c1_A + c1_B + c1_C
        c2 = c2_A + c2_B + c2_C
        c3 = c3_A + c3_B + c3_C
        s1 = s1_A + s1_B + s1_C
        s2 = s2_A + s2_B + s2_C

        return c0, c1, c2, c3, s1, s2

    def coeffs_plus_plus_UP(self, ReCC, ReCC_V, ReCC_A, ImCC, ImCC_V, ImCC_A):
        """Computes helicity-conserving unpolarised target coefficients."""
        Q2, t, x, e2, y, K, t_min, K_tilde, helc = self.Q2, self.t, self.x, self.e2, self.y, self.K, self.t_min, self.K_tilde, self.helc

        EE = (1.+e2)*(1.+e2)
        RE = tf.sqrt(1.+e2)
        RE5 = tf.pow(RE, 5.)
        TQ = t/Q2
        KTQ = K_tilde*K_tilde/Q2
        A = (1.-y-e2*y*y/4.)
        tp = t-t_min
        TPQ = tp/Q2

        c0_pp = -(4.*(2.-y)*(1.+RE)/EE) * \
            (KTQ*(2.-y)*(2.-y)/RE + TQ*A*(2.-x) *
             (1.+(2.*x*(2.-x+0.5*(RE-1.)+0.5*e2/x)*TQ+e2)/((2.-x)*(1.+RE))))

        c0_pp_V = (8.*(2.-y)*x*TQ/EE) * \
            ((2.-y)*(2.-y)*(KTQ/RE) + A*0.5*(1.+RE)
             * (1.+TQ)*(1.+(RE-1.+2.*x)*TQ/(1.+RE)))

        c0_pp_A = (8.*(2.-y)*TQ/EE) * \
            (KTQ*(2.-y)*(2.-y)*0.5*(1.+RE-2.*x)/RE +
             A*(0.5*(1.+RE) * (1.+RE-x+(RE-1.+x*(3.+RE-2.*x)/(1.+RE))*TQ) - 2.*KTQ))

        c1_pp = -(16.*K*A/RE5)*((1.+(1.-x)*0.5*(RE-1.)/x+0.25*e2/x)*x*TQ-0.25*3.*e2) \
            - 4.*K*(2.-2.*y+y*y+0.5*e2*y*y)*((1.+RE-e2)/RE5) * \
            (1.-(1.-3.*x)*TQ + (1.-RE+3.*e2)*x*TQ/(1.+RE-e2))

        c1_pp_V = (16.*K*x*TQ/RE5) * \
            ((2.-y)*(2.-y)*(1.-(1.-2.*x)*TQ)+A*(1.+RE-2.*x)*TPQ*0.5)

        c1_pp_A = -(16.*K*TQ/EE) * \
            (A*(1.-(1.-2.*x)*TQ+(4.*x*(1.-x)+e2)*TPQ/(4.*RE)) - (2.-y)*(2.-y) *
             (1.-0.5*x+0.25*(1.+RE-2.*x)*(1.-TQ) + (4.*x*(1.-x)+e2)*TPQ/(2.*RE)))

        c2_pp = (8.*(2.-y)*A/EE) * \
            (2.*e2*KTQ/(RE*(1.+RE)) + x*TQ*TPQ*(1.-x-0.5*(RE-1.)+0.5*e2/x))

        c2_pp_V = (8.*(2.-y)*A/EE)*x*TQ * \
            (4.*KTQ/RE + 0.5*(1.+RE-2.*x)*(1.+TQ)*TPQ)

        c2_pp_A = (4.*(2.-y)*A/EE)*TQ * \
            (4.*(1.-2.*x)*KTQ/RE - (3.-RE-2.*x+e2/x)*x*TPQ)

        c3_pp = -(8.*K*A*(RE-1.)/RE5)*((1.-x)*TQ+0.5*(RE-1.)*(1.+TQ))

        c3_pp_V = -(8.*K*A*x*TQ/RE5)*(RE-1.+(1.+RE-2.*x)*TQ)

        c3_pp_A = (16.*K*A*TQ*TPQ/RE5)*(x*(1.-x)+0.25*e2)

        s1_pp = helc * (8.*K*(2.-y)*y/(1.+e2)) * \
            (1.+(1.-x+0.5*(RE-1))*TPQ/(1.+e2))

        s1_pp_V = -helc * (8.*K*(2.-y)*y*x*TQ/EE) * (RE-1.+(1.+RE-2.*x)*TQ)

        s1_pp_A = helc * (8.*K*(2.-y)*y*TQ/(1.+e2)) * \
            (1.-(1.-2.*x)*(1.+RE-2.*x)*TPQ*0.5/RE)

        s2_pp = -helc * (4.*A*y*(1.+RE-2.*x)*TPQ/(tf.pow(RE, 3.))) * \
            ((e2-x*(RE-1.))/(1.+RE-2.*x) - (2.*x+e2)*TPQ/(2.*RE))

        s2_pp_V = -helc * (4.*A*y*x*TQ*(1.-(1.-2.*x)*TQ)/EE) * \
            (RE-1.+(1.+RE-2.*x)*TQ)

        s2_pp_A = -helc * (8.*A*y*TQ*TPQ*(1.+RE-2.*x)/EE) * \
            (1.+(4.*(1.-x)*x+e2)*TQ/(4.-2.*x+3.*e2))

        # BKM 2010 : Eq (2.36)
        CC0_pp = ReCC + c0_pp_V*ReCC_V/c0_pp + c0_pp_A*ReCC_A/c0_pp
        CC1_pp = ReCC + c1_pp_V*ReCC_V/c1_pp + c1_pp_A*ReCC_A/c1_pp
        CC2_pp = ReCC + c2_pp_V*ReCC_V/c2_pp + c2_pp_A*ReCC_A/c2_pp
        CC3_pp = ReCC + c3_pp_V*ReCC_V/c3_pp + c3_pp_A*ReCC_A/c3_pp

        SS1_pp = tf.where(tf.not_equal(s1_pp, 0.0),
                          ImCC + s1_pp_V*ImCC_V/s1_pp + s1_pp_A*ImCC_A/s1_pp,
                          ImCC)
        SS2_pp = tf.where(tf.not_equal(s2_pp, 0.0),
                          ImCC + s2_pp_V*ImCC_V/s2_pp + s2_pp_A*ImCC_A/s2_pp,
                          ImCC)

        c0 = c0_pp*CC0_pp
        c1 = c1_pp*CC1_pp
        c2 = c2_pp*CC2_pp
        c3 = c3_pp*CC3_pp
        s1 = s1_pp*SS1_pp
        s2 = s2_pp*SS2_pp

        return c0, c1, c2, c3, s1, s2

    def coeffs_zero_plus_UP(self, ReCC_eff, ReCC_V_eff, ReCC_A_eff, ImCC_eff, ImCC_V_eff, ImCC_A_eff):
        """Computes longitudinal-transverse unpolarised target helicity-changing coefficients."""
        Q2, t, x, e2, y, K, t_min, K_tilde, helc = self.Q2, self.t, self.x, self.e2, self.y, self.K, self.t_min, self.K_tilde, self.helc

        EE = (1.+e2)*(1.+e2)
        RE = tf.sqrt(1.+e2)
        RE5 = tf.pow(RE, 5.)
        TQ = t/Q2
        KTQ = K_tilde*K_tilde/Q2
        A = (1.-y-e2*y*y/4.)
        tp = t-t_min
        TPQ = tp/Q2

        c0_0p = (12.*tf.sqrt(2.*A)*K*(2.-y)/RE5) * (e2+(2.-6.*x-e2)*TQ/3.)

        c1_0p = (8.*tf.sqrt(2.*A)/EE) * \
            ((2.-y)*(2.-y)*TPQ*(1.-x+((1.-x)*x+0.25*e2)*TPQ/RE) +
             (A/RE)*(1.-(1.-2.*x)*TQ)*(e2-2.*(1.+0.5*e2/x)*x*TQ))

        c2_0p = -(8.*tf.sqrt(2.*A)*K*(2.-y)*(1.+0.5*e2)/RE5) * \
            (1. + (1. + 0.5*e2/x)*x*TQ/(1.+0.5*e2))

        s1_0p = helc*8.*tf.sqrt(2.*A)*(2.-y)*y*KTQ/EE

        s2_0p = helc*(8.*tf.sqrt(2.*A)*K*y*(1.+0.5*e2)/EE) * \
            (1. + (1.+0.5*e2/x)*x*TQ/(1.+0.5*e2))

        c0_0p_V = (24.*tf.sqrt(2.*A)*K*(2.-y)*x*TQ/RE5)*(1.-(1.-2.*x)*TQ)

        c1_0p_V = (16.*tf.sqrt(2.*A)*x*TQ/RE5) * \
            (KTQ*(2.-y)*(2.-y) + (1.-(1.-2.*x)*TQ)*(1.-(1.-2.*x)*TQ)*A)

        c2_0p_V = c0_0p_V/3.

        s1_0p_V = helc*(4.*tf.sqrt(2.*A)*y*(2.-y)*x*TQ/EE) * \
            (4.*(1.-2.*x)*TQ*(1.+x*TQ)+e2*(1.+TQ)*(1.+TQ))

        s2_0p_V = -helc*(8.*tf.sqrt(2.*A)*K*y*x*TQ/EE)*(1.-(1.-2.*x)*TQ)

        c0_0p_A = (4.*tf.sqrt(2.*A)*K*(2.-y)*TQ*(8.-6.*x+5.*e2)/RE5) * \
            (1.-TQ*(2.-12.*x*(1.-x)-e2)/(8.-6.*x+5.*e2))

        c1_0p_A = (8.*tf.sqrt(2.*A)*TQ/RE5) * \
            (KTQ*(1.-2.*x)*(2.-y)*(2.-y)+(1.-(1.-2.*x)*TQ)
             * A*(4.-2.*x+3.*e2+TQ*(4.*x*(1.-x)+e2)))

        c2_0p_A = (8.*tf.sqrt(2.*A)*K*(2.-y)*TQ/EE) * \
            (1.-x+0.5*TPQ*(4.*x*(1.-x)+e2)/RE)

        s1_0p_A = -helc*8.*tf.sqrt(2.*A)*y*(2.-y)*(1.-2.*x)*TQ*K*K/EE

        s2_0p_A = -helc*(2.*tf.sqrt(2.*A)*K*y*TQ/EE) * \
            (4.-4.*x+2.*e2+4.*TQ*(4.*x*(1.-x)+e2))

        # BKM 2010 : Eq (2.37)
        B = tf.sqrt(2.)*KTQ/(2.-x)

        CC0_0p = B*(ReCC_eff + c0_0p_V*ReCC_V_eff /
                    c0_0p + c0_0p_A*ReCC_A_eff/c0_0p)
        CC1_0p = B*(ReCC_eff + c1_0p_V*ReCC_V_eff /
                    c1_0p + c1_0p_A*ReCC_A_eff/c1_0p)
        CC2_0p = B*(ReCC_eff + c2_0p_V*ReCC_V_eff /
                    c2_0p + c2_0p_A*ReCC_A_eff/c2_0p)

        SS1_0p = tf.where(tf.not_equal(s1_0p, 0.0),
                          B*ImCC_eff + s1_0p_V*ImCC_V_eff/s1_0p + s1_0p_A*ImCC_A_eff/s1_0p,
                          B*ImCC_eff)
        SS2_0p = tf.where(tf.not_equal(s2_0p, 0.0),
                          B*ImCC_eff + s2_0p_V*ImCC_V_eff/s2_0p + s2_0p_A*ImCC_A_eff/s2_0p,
                          B*ImCC_eff)

        c0 = c0_0p*CC0_0p
        c1 = c1_0p*CC1_0p
        c2 = c2_0p*CC2_0p
        c3 = 0.
        s1 = s1_0p*SS1_0p
        s2 = s2_0p*SS2_0p

        return c0, c1, c2, c3, s1, s2


def compute_quantities(k, Q2, x, t, phi, ReH, ReHt, ReE, ReEt, ImH, ImHt, ImE, ImEt):
    """
    Compute differential cross section quantities using BKM10 formalism.
    
    Parameters:
        k, Q2, x, t, phi: Kinematic variables
        ReH, ReHt, ReE, ReEt, ImH, ImHt, ImE, ImEt: CFF values
    
    Returns:
        dsig, BH, DVCS, I: Differential cross section and components
    """
    # Force everything to float32 to match the layer's constants and math
    to32 = lambda v: tf.convert_to_tensor(v, dtype=tf.float32)

    k   = to32(k)
    Q2  = to32(Q2)
    x   = to32(x)
    t   = to32(t)
    phi = to32(phi)

    cffs = tf.stack([
        to32(ReH), to32(ReHt), to32(ReE), to32(ReEt),
        to32(ImH), to32(ImHt), to32(ImE), to32(ImEt)
    ], axis=0)[tf.newaxis, :]  # shape (1, 8), dtype float32

    dsig, BH, DVCS, I = BKM10(k, Q2, x, t).diff_cross(phi, cffs)
    return dsig, BH, DVCS, I


# ============================================================================
# From km15.py
# ============================================================================

# KM15 parameters
nval = 1.35
pval = 1.
nsea = 1.5
rsea = 1.
psea = 2.
bsea = 4.6
Mval = 0.789
rval = 0.918
bval = 0.4
C0 = 2.768
Msub = 1.204
Mtval = 3.993
rtval = 0.881
btval = 0.4
ntval = 0.6
Msea = sqrt(0.482)
rpi = 2.646
Mpi = 4.


def ModKM15_CFFs(QQ, xB, t, k=0.0):
    """
    Compute KM15 CFFs (Compton Form Factors) for given kinematics.
    
    Parameters:
        QQ: Q² (photon virtuality in GeV²)
        xB: Bjorken x
        t: momentum transfer squared (GeV²)
        k: incoming lepton energy (GeV), optional
    
    Returns:
        ReH, ImH, ReE, ReHtilde, ImHtilde, ReEtilde: CFF values
    """
    alpha_val = 0.43 + 0.85 * t
    alpha_sea = 1.13 + 0.15 * t
    Ct = C0 / pow(1. - t / Msub / Msub, 2.)
    xi = xB / (2. - xB)

    def fHval(x):
        return (nval * rval) / (1. + x) * pow((2. * x) / (1. + x), -alpha_val) * \
               pow((1. - x) / (1. + x), bval) * \
               1. / pow(1. - ((1. - x) / (1. + x)) * (t / Mval / Mval), pval)

    def fHsea(x):
        return (nsea * rsea) / (1. + x) * pow((2. * x) / (1. + x), -alpha_sea) * \
               pow((1. - x) / (1. + x), bsea) * \
               1. / pow(1. - ((1. - x) / (1. + x)) * (t / Msea / Msea), psea)

    def fHtval(x):
        return (ntval * rtval) / (1. + x) * pow((2. * x) / (1. + x), -alpha_val) * \
               pow((1. - x) / (1. + x), btval) * \
               1. / (1. - ((1. - x) / (1. + x)) * (t / Mtval / Mtval))

    def fImH(x):
        return pi * ((2. * (4. / 9.) + 1. / 9.) * fHval(x) + 2. / 9. * fHsea(x))

    def fImHt(x):
        return pi * (2. * (4. / 9.) + 1. / 9.) * fHtval(x)

    def fPV_ReH(x):
        return -2. * x / (x + xi) * fImH(x)

    def fPV_ReHt(x):
        return -2. * xi / (x + xi) * fImHt(x)

    # Principal value integrals
    DR_ReH, _ = quad(fPV_ReH, 1e-6, 1.0, weight='cauchy', wvar=xi)
    DR_ReHt, _ = quad(fPV_ReHt, 1e-6, 1.0, weight='cauchy', wvar=xi)

    # Evaluate the CFFs
    ImH = fImH(xi)
    ReH = 1. / pi * DR_ReH - Ct
    ReE = Ct
    ImHtilde = fImHt(xi)
    ReHtilde = 1. / pi * DR_ReHt
    ReEtilde = rpi / xi * 2.164 / ((0.0196 - t) * pow(1. - t / Mpi / Mpi, 2.))

    return ReH, ImH, ReE, ReHtilde, ImHtilde, ReEtilde

