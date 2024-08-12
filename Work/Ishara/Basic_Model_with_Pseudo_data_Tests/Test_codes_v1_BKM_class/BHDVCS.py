import numpy as np
import math

class BHDVCS(object):

    def __init__(self):
        self.ALP_INV = 137.0359998  # 1 / Electromagnetic Fine Structure Constant
        self.PI = 3.1415926535
        self.RAD = self.PI / 180.0
        self.M = 0.938272  # Mass of the proton in GeV
        self.GeV2nb = .389379 * 1000000  # Conversion from GeV to NanoBar
        self.M2 = 0.938272 * 0.938272  # Mass of the proton squared in GeV

    def SetKinematics(self, QQ, x, t, k):
        ee = 4.0 * self.M2 * x * x / QQ  # epsilon squared
        y = np.sqrt(QQ) / (np.sqrt(ee) * k)  # lepton energy fraction
        xi = x * (1.0 + t / 2.0 / QQ) / (2.0 - x + x * t / QQ)  # Generalized Bjorken variable
        Gamma = x * y * y / self.ALP_INV**3 / self.PI / 8.0 / QQ / QQ / np.sqrt(1.0 + ee)  # factor in front of the cross section, eq. (22)
        tmin = - QQ * (2.0 * (1.0 - x) * (1.0 - np.sqrt(1.0 + ee)) + ee) / (4.0 * x * (1.0 - x) + ee)  # eq. (31)
        Ktilde_10 = np.sqrt(tmin - t) * np.sqrt((1.0 - x) * np.sqrt(1.0 + ee) + ((t - tmin) * (ee + 4.0 * x * (1.0 - x)) / 4.0 / QQ)) * np.sqrt(1.0 - y - y * y * ee / 4.0) / np.sqrt(1.0 - y + y * y * ee / 4.0)  # K tilde from 2010 paper
        K = np.sqrt(1.0 - y + y * y * ee / 4.0) * Ktilde_10 / np.sqrt(QQ)
        return ee, y, xi, Gamma, tmin, Ktilde_10, K

    def BHLeptonPropagators(self, phi, QQ, x, t, ee, y, K):
        KD = - QQ / (2.0 * y * (1.0 + ee)) * (1.0 + 2.0 * K * np.cos(self.PI - (phi * self.RAD)) - t / QQ * (1.0 - x * (2.0 - y) + y * ee / 2.0) + y * ee / 2.0)  # eq. (29)

        P1 = 1.0 + 2.0 * KD / QQ
        P2 = t / QQ - 2.0 * KD / QQ
        return P1, P2

    def BHUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K):
        phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K = [np.float32(i) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K]]
        c0_BH = 8.0 * K * K * ((2.0 + 3.0 * ee) * (QQ / t) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) + 2.0 * x * x * (F1 + F2) * (F1 + F2)) + (2.0 - y) * (2.0 - y) * ((2.0 + ee) * ((4.0 * x * x * self.M2 / t) * (1.0 + t / QQ) * (1.0 + t / QQ) + 4.0 * (1.0 - x) * (1.0 + x * t / QQ)) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) + 4.0 * x * x * (x + (1.0 - x + ee / 2.0) * (1.0 - t / QQ) * (1.0 - t / QQ) - x * (1.0 - 2.0 * x) * t * t / (QQ * QQ)) * (F1 + F2) * (F1 + F2)) + 8.0 * (1.0 + ee) * (1.0 - y - ee * y * y / 4.0) * (2.0 * ee * (1.0 - t / (4.0 * self.M2)) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) - x * x * (1.0 - t / QQ) * (1.0 - t / QQ) * (F1 + F2) * (F1 + F2))

        c1_BH = 8.0 * K * (2.0 - y) * ((4.0 * x * x * self.M2 / t - 2.0 * x - ee) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) + 2.0 * x * x * (1.0 - (1.0 - 2.0 * x) * t / QQ) * (F1 + F2) * (F1 + F2))

        c2_BH = 8.0 * x * x * K * K * ((4.0 * self.M2 / t) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) + 2.0 * (F1 + F2) * (F1 + F2))

        Amp2_BH = 1.0 / (x * x * y * y * (1.0 + ee) * (1.0 + ee) * t * P1 * P2) * (c0_BH + c1_BH * np.cos(self.PI - (phi * self.RAD)) + c2_BH * np.cos(2.0 * (self.PI - (phi * self.RAD))))

        Amp2_BH = self.GeV2nb * Amp2_BH  # conversion to nb

        return Gamma * Amp2_BH

    def IUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, twist, tmin, xi, Ktilde_10):
        phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10 = [np.float32(i) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10]]
        self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)

        A_U_I, B_U_I, C_U_I = self.ABC_UU_I_10(phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10)

        I_10 = 1.0 / (x * y * y * y * t * P1 * P2) * (A_U_I * (F1 * ReH - t / 4.0 / self.M2 * F2 * ReE) + B_U_I * (F1 + F2) * (ReH + ReE) + C_U_I * (F1 + F2) * ReHtilde)

        I_10 = self.GeV2nb * I_10  # conversion to nb

        return Gamma * I_10

    def ABC_UU_I_10(self, phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10):
        if twist == "t2":
            f = 0  # F_eff = 0 (pure twist 2)
        elif twist == "t3":
            f = -2.0 * xi / (1.0 + xi)
        elif twist == "t3ww":
            f = 2.0 / (1.0 + xi)

        # Define C_110, C_110_V, C_110_A
        C_110 = -4.0 * (2.0 - y) * (1.0 + np.sqrt(1 + ee)) / (1.0 + ee)**2 * (Ktilde_10**2 * (2.0 - y)**2 / QQ / np.sqrt(1 + ee) + t / QQ * (1.0 - y - ee / 4.0 * y * y) * (2.0 - x) * (1.0 + (2.0 * x * (2.0 - x + (np.sqrt(1.0 + ee) - 1.0) / 2.0 + ee / 2.0 / x) * t / QQ + ee) / (2.0 - x) / (1.0 + np.sqrt(1.0 + ee))))
        C_110_V = 8.0 * (2.0 - y) / (1.0 + ee)**2 * x * t / QQ * ((2.0 - y) * (2.0 - y) / np.sqrt(1.0 + ee) * Ktilde_10 * Ktilde_10 / QQ + (1.0 - y - ee / 4.0 * y * y) * (1.0 + np.sqrt(1.0 + ee)) / 2.0 * (1.0 + t / QQ) * (1.0 + (np.sqrt(1.0 + ee) - 1.0 + 2.0 * x) / (1.0 + np.sqrt(1.0 + ee)) * t / QQ))
        C_110_A = 8.0 * (2.0 - y) / (1.0 + ee)**2 * t / QQ * ((2.0 - y) * (2.0 - y) / np.sqrt(1.0 + ee) * Ktilde_10 * Ktilde_10 / QQ * (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) / 2.0 + (1.0 - y - ee / 4.0 * y * y) * ((1.0 + np.sqrt(1.0 + ee)) / 2.0 * (1.0 + np.sqrt(1.0 + ee) - x + (np.sqrt(1.0 + ee) - 1.0 + x * (3.0 + np.sqrt(1.0 + ee) - 2.0 * x) / (1.0 + np.sqrt(1.0 + ee))) * t / QQ) - 2.0 * Ktilde_10 * Ktilde_10 / QQ))

        # Define C_010, C_010_V, C_010_A
        C_010 = 12.0 * math.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * (ee + (2.0 - 6.0 * x - ee) / 3.0 * t / QQ)
        C_010_V = 24.0 * math.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * x * t / QQ * (1.0 - (1.0 - 2.0 * x) * t / QQ)
        C_010_A = 4.0 * math.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * t / QQ * (8.0 - 6.0 * x + 5.0 * ee) * (1.0 - t / QQ * ((2.0 - 12.0 * x * (1.0 - x) - ee) / (8.0 - 6.0 * x + 5.0 * ee)))

        # Define C_111, C_111_V, C_111_A
        C_111 = -16.0 * K * (1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * ((1.0 + (1.0 - x) * (np.sqrt(1.0 + ee) - 1.0) / 2.0 / x + ee / 4.0 / x) * x * t / QQ - 3.0 * ee / 4.0) - 4.0 * K * (2.0 - 2.0 * y + y * y + ee / 2.0 * y * y) * (1.0 + np.sqrt(1.0 + ee) - ee) / (np.sqrt(1.0 + ee))**5 * (1.0 - (1.0 - 3.0 * x) * t / QQ + (1.0 - np.sqrt(1.0 + ee) + 3.0 * ee) / (1.0 + np.sqrt(1.0 + ee) - ee) * x * t / QQ)
        C_111_V = 16.0 * K / (np.sqrt(1.0 + ee))**5 * x * t / QQ * ((2.0 - y) * (2.0 - y) * (1.0 - (1.0 - 2.0 * x) * t / QQ) + (1.0 - y - ee / 4.0 * y * y) * (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) / 2.0 * (t - tmin) / QQ)
        C_111_A = -16.0 * K / (1.0 + ee)**2 * t / QQ * ((1.0 - y - ee / 4.0 * y * y) * (1.0 - (1.0 - 2.0 * x) * t / QQ + (4.0 * x * (1.0 - x) + ee) / 4.0 / np.sqrt(1.0 + ee) * (t - tmin) / QQ) - (2.0 - y)**2 * (1.0 - x / 2.0 + (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) / 4.0 * (1.0 - t / QQ) + (4.0 * x * (1.0 - x) + ee) / 2.0 / np.sqrt(1.0 + ee) * (t - tmin) / QQ))

        # Define C_112, C_112_V, C_112_A
        C_112 = 8.0 * (2.0 - y) * (1.0 - y - ee / 4.0 * y * y) / (1.0 + ee)**2 * (2.0 * ee / np.sqrt(1.0 + ee) / (1.0 + np.sqrt(1.0 + ee)) * Ktilde_10**2 / QQ + x * t * (t - tmin) / QQ / QQ * (1.0 - x - (np.sqrt(1.0 + ee) - 1.0) / 2.0 + ee / 2.0 / x))
        C_112_V = 8.0 * (2.0 - y) * (1.0 - y - ee / 4.0 * y * y) / (1.0 + ee)**2 * x * t / QQ * (4.0 * Ktilde_10**2 / np.sqrt(1.0 + ee) / QQ + (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) / 2.0 * (1.0 + t / QQ) * (t - tmin) / QQ)
        C_112_A = 4.0 * (2.0 - y) * (1.0 - y - ee / 4.0 * y * y) / (1.0 + ee)**2 * t / QQ * (4.0 * (1.0 - 2.0 * x) * Ktilde_10**2 / np.sqrt(1.0 + ee) / QQ - (3.0 - np.sqrt(1.0 + ee) - 2.0 * x + ee / x) * x * (t - tmin) / QQ)

        # Define C_113, C_113_V, C_113_A
        C_113 = -8.0 * K * (1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * (np.sqrt(1.0 + ee) - 1.0) * ((1.0 - x) * t / QQ + (np.sqrt(1.0 + ee) - 1.0) / 2.0 * (1.0 + t / QQ))
        C_113_V = -8.0 * K * (1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * x * t / QQ * (np.sqrt(1.0 + ee) - 1.0 + (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) * t / QQ)
        C_113_A = 16.0 * K * (1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * t * (t - tmin) / QQ / QQ * (x * (1.0 - x) + ee / 4.0)

        # Define C_012, C_012_V, C_012_A
        C_012 = -8.0 * math.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * (1.0 + ee / 2.0) * (1.0 + (1.0 + ee / 2.0 / x) / (1.0 + ee / 2.0) * x * t / QQ)
        C_012_V = 8.0 * math.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * x * t / QQ * (1.0 - (1.0 - 2.0 * x) * t / QQ)
        C_012_A = 8.0 * math.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (1.0 + ee)**2 * t / QQ * (1.0 - x + (t - tmin) / 2.0 / QQ * (4.0 * x * (1.0 - x) + ee) / np.sqrt(1.0 + ee))

        # Define C_011, C_011_V, C_011_A
        C_011 = 8.0 * math.sqrt(2.0) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (1.0 + ee)**2 * (2.0 * (1.0 - x) * t / QQ + (1.0 + (1.0 - x) * (np.sqrt(1.0 + ee) - 1.0)) * (t - tmin) / QQ)  # Helicty-changing (F_eff) interference term
        C_011_V = 16.0 * math.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee))**5 * x * t / QQ * (1.0 - (1.0 - 2.0 * x) * t / QQ)
        C_011_A = 8.0 * math.sqrt(2.0) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (1.0 + ee)**2 * t / QQ * (4.0 * (1.0 - x) * Ktilde_10**2 / np.sqrt(1.0 + ee) / QQ + (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) * t / QQ)

        A_U_I = C_110 + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_010 + (C_111 + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_011) * np.cos(self.PI - (phi * self.RAD)) + (C_112 + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_012) * np.cos(2.0 * (self.PI - (phi * self.RAD))) + C_113 * np.cos(3.0 * (self.PI - (phi * self.RAD)))
        B_U_I = xi / (1.0 + t / 2.0 / QQ) * (C_110_V + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_010_V + (C_111_V + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_011_V) * np.cos(self.PI - (phi * self.RAD)) + (C_112_V + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_012_V) * np.cos(2.0 * (self.PI - (phi * self.RAD))) + C_113_V * np.cos(3.0 * (self.PI - (phi * self.RAD))))
        C_U_I = xi / (1.0 + t / 2.0 / QQ) * (C_110 + C_110_A + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * (C_010 + C_010_A) + (C_111 + C_111_A + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * (C_011 + C_011_A)) * np.cos(self.PI - (phi * self.RAD)) + (C_112 + C_112_A + math.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * (C_012 + C_012_A)) * np.cos(2.0 * (self.PI - (phi * self.RAD))) + (C_113 + C_113_A) * np.cos(3.0 * (self.PI - (phi * self.RAD))))

        return A_U_I, B_U_I, C_U_I

class F_calc:
    def __init__(self):
        self.module = BHDVCS()

    def fn_1(self, kins, cffs):
        phi, QQ, x, t, k, F1, F2 = kins
        ReH, ReE, ReHtilde, c0fit = cffs
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.module.SetKinematics(QQ, x, t, k)
        P1, P2 = self.module.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.module.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.module.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return f_pred  # Directly return the value

class F1F2:
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
