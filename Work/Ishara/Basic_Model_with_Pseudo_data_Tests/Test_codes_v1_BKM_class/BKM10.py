import numpy as np
import math

class BHDVCStf:
    def __init__(self):
        self.ALP_INV = 137.0359998  # 1 / Electromagnetic Fine Structure Constant
        self.PI = 3.1415926535
        self.RAD = self.PI / 180.0
        self.M = 0.938272  # Mass of the proton in GeV
        self.GeV2nb = 0.389379 * 1e6  # Conversion from GeV to NanoBar
        self.M2 = self.M ** 2  # Mass of the proton squared in GeV

    def set_kinematics(self, QQ, x, t, k):
        ee = 4.0 * self.M2 * x * x / QQ  # epsilon squared
        y = np.sqrt(QQ) / (np.sqrt(ee) * k)  # lepton energy fraction
        xi = x * (1.0 + t / 2.0 / QQ) / (2.0 - x + x * t / QQ)  # Generalized Bjorken variable
        Gamma = x * y * y / self.ALP_INV ** 3 / self.PI / 8.0 / QQ / QQ / np.sqrt(1.0 + ee)  # factor in front of the cross section, eq. (22)
        tmin = -QQ * (2.0 * (1.0 - x) * (1.0 - np.sqrt(1.0 + ee)) + ee) / (4.0 * x * (1.0 - x) + ee)  # eq. (31)
        Ktilde_10 = (np.sqrt(tmin - t) *
                     np.sqrt((1.0 - x) * np.sqrt(1.0 + ee) + ((t - tmin) * (ee + 4.0 * x * (1.0 - x)) / 4.0 / QQ)) *
                     np.sqrt(1.0 - y - y * y * ee / 4.0) / np.sqrt(1.0 - y + y * y * ee / 4.0))  # K tilde from 2010 paper
        K = np.sqrt(1.0 - y + y * y * ee / 4.0) * Ktilde_10 / np.sqrt(QQ)
        return ee, y, xi, Gamma, tmin, Ktilde_10, K

    def bh_lepton_propagators(self, phi, QQ, x, t, ee, y, K):
        # KD 4-vector product (phi-dependent)
        KD = -QQ / (2.0 * y * (1.0 + ee)) * (1.0 + 2.0 * K * np.cos(self.PI - (phi * self.RAD)) -
                                              t / QQ * (1.0 - x * (2.0 - y) + y * ee / 2.0) + y * ee / 2.0)  # eq. (29)

        # lepton BH propagators P1 and P2 (contaminating phi-dependence)
        P1 = 1.0 + 2.0 * KD / QQ
        P2 = t / QQ - 2.0 * KD / QQ
        return P1, P2

    def bh_uu(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K):
        # BH unpolarized Fourier harmonics eqs. (35 - 37)
        c0_BH = (8.0 * K * K * ((2.0 + 3.0 * ee) * (QQ / t) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) +
                                2.0 * x * x * (F1 + F2) * (F1 + F2)) +
                 (2.0 - y) * (2.0 - y) * ((2.0 + ee) * ((4.0 * x * x * self.M2 / t) * (1.0 + t / QQ) * (1.0 + t / QQ) +
                                                       4.0 * (1.0 - x) * (1.0 + x * t / QQ)) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) +
                                      4.0 * x * x * (x + (1.0 - x + ee / 2.0) * (1.0 - t / QQ) * (1.0 - t / QQ) -
                                                     x * (1.0 - 2.0 * x) * t * t / (QQ * QQ)) * (F1 + F2) * (F1 + F2)) +
                 8.0 * (1.0 + ee) * (1.0 - y - ee * y * y / 4.0) * (2.0 * ee * (1.0 - t / (4.0 * self.M2)) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) -
                                                                     x * x * (1.0 - t / QQ) * (1.0 - t / QQ) * (F1 + F2) * (F1 + F2)))

        c1_BH = 8.0 * K * (2.0 - y) * ((4.0 * x * x * self.M2 / t - 2.0 * x - ee) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) +
                                       2.0 * x * x * (1.0 - (1.0 - 2.0 * x) * t / QQ) * (F1 + F2) * (F1 + F2))

        c2_BH = 8.0 * x * x * K * K * ((4.0 * self.M2 / t) * (F1 * F1 - F2 * F2 * t / (4.0 * self.M2)) + 2.0 * (F1 + F2) * (F1 + F2))

        # BH squared amplitude eq (25) divided by e^6
        Amp2_BH = (1.0 / (x * x * y * y * (1.0 + ee) * (1.0 + ee) * t * P1 * P2) *
                   (c0_BH + c1_BH * np.cos(self.PI - (phi * self.RAD)) + c2_BH * np.cos(2.0 * (self.PI - (phi * self.RAD)))))

        Amp2_BH = self.GeV2nb * Amp2_BH  # conversion to nb

        return Gamma * Amp2_BH

    def i_uu(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, twist, tmin, xi, Ktilde_10):
        # Get BH propagators and set the kinematics
        self.bh_lepton_propagators(phi, QQ, x, t, ee, y, K)

        # Get A_UU_I, B_UU_I and C_UU_I interference coefficients
        A_U_I, B_U_I, C_U_I = self.abc_uu_i_10(phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10)

        # BH-DVCS interference squared amplitude
        I_10 = (1.0 / (x * y * y * y * t * P1 * P2) *
                (A_U_I * (F1 * ReH - t / 4.0 / self.M2 * F2 * ReE) + B_U_I * (F1 + F2) * (ReH + ReE) +
                 C_U_I * (F1 + F2) * ReHtilde))

        I_10 = self.GeV2nb * I_10  # conversion to nb

        return Gamma * I_10

    def abc_uu_i_10(self, phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10):
        # Interference coefficients  (BKM10 Appendix A.1)
        if twist == "t2":
            f = 0  # F_eff = 0 (pure twist 2)
        elif twist == "t3":
            f = -2.0 * xi / (1.0 + xi)
        elif twist == "t3ww":
            f = 2.0 / (1.0 + xi)

        # helicity - conserving (F)
        C_110 = -4.0 * (2.0 - y) * (1.0 + np.sqrt(1 + ee)) / (1.0 + ee) ** 2 * (
                Ktilde_10 ** 2 * (2.0 - y) ** 2 / QQ / np.sqrt(1 + ee)
                + t / QQ * (1.0 - y - ee / 4.0 * y * y) * (2.0 - x) * (1.0 + (
                2.0 * x * (2.0 - x + (np.sqrt(1.0 + ee) - 1.0) / 2.0 + ee / 2.0 / x) * t / QQ + ee) /
                                                                     (2.0 - x) / (1.0 + np.sqrt(1.0 + ee))))
        C_110_V = 8.0 * (2.0 - y) / (1.0 + ee) ** 2 * x * t / QQ * (
                (2.0 - y) ** 2 / np.sqrt(1.0 + ee) * Ktilde_10 ** 2 / QQ
                + (1.0 - y - ee / 4.0 * y * y) * (1.0 + np.sqrt(1.0 + ee)) / 2.0 * (1.0 + t / QQ) * (
                1.0 + (np.sqrt(1.0 + ee) - 1.0 + 2.0 * x) / (1.0 + np.sqrt(1.0 + ee)) * t / QQ))
        C_110_A = 8.0 * (2.0 - y) / (1.0 + ee) ** 2 * t / QQ * (
                (2.0 - y) ** 2 / np.sqrt(1.0 + ee) * Ktilde_10 ** 2 / QQ * (
                1.0 + np.sqrt(1.0 + ee) - 2.0 * x) / 2.0
                + (1.0 - y - ee / 4.0 * y * y) * ((1.0 + np.sqrt(1.0 + ee)) / 2.0 * (
                1.0 + np.sqrt(1.0 + ee) - x + (
                np.sqrt(1.0 + ee) - 1.0 + x * (3.0 + np.sqrt(1.0 + ee) - 2.0 * x) / (1.0 + np.sqrt(1.0 + ee)))
                * t / QQ) - 2.0 * Ktilde_10 ** 2 / QQ))
        # helicity - changing (F_eff)
        C_010 = 12.0 * np.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * (
                ee + (2.0 - 6.0 * x - ee) / 3.0 * t / QQ)
        C_010_V = 24.0 * np.sqrt(2.0) * K * (2.0 - y) * np.sqrt(
            1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * x * t / QQ * (1.0 - (1.0 - 2.0 * x) * t / QQ)
        C_010_A = 4.0 * np.sqrt(2.0) * K * (2.0 - y) * np.sqrt(
            1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * t / QQ * (
                              8.0 - 6.0 * x + 5.0 * ee) * (
                              1.0 - t / QQ * ((2.0 - 12 * x * (1.0 - x) - ee) / (8.0 - 6.0 * x + 5.0 * ee)))
        # n = 1 -----------------------------------------
        # helicity - conserving (F)
        C_111 = -16.0 * K * (1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * (
                (1.0 + (1.0 - x) * (np.sqrt(1 + ee) - 1.0) / 2.0 / x + ee / 4.0 / x) * x * t / QQ - 3.0 * ee / 4.0) - 4.0 * K * (
                2.0 - 2.0 * y + y * y + ee / 2.0 * y * y) * (1.0 + np.sqrt(1 + ee) - ee) / (np.sqrt(1.0 + ee) ** 5) * (
                1.0 - (1.0 - 3.0 * x) * t / QQ + (
                1.0 - np.sqrt(1 + ee) + 3.0 * ee) / (1.0 + np.sqrt(1 + ee) - ee) * x * t / QQ)
        C_111_V = 16.0 * K / (np.sqrt(1.0 + ee) ** 5) * x * t / QQ * (
                (2.0 - y) ** 2 * (1.0 - (1.0 - 2.0 * x) * t / QQ) + (1.0 - y - ee / 4.0 * y * y)
                * (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) / 2.0 * (t - tmin) / QQ)
        C_111_A = -16.0 * K / (1.0 + ee) ** 2 * t / QQ * (
                (1.0 - y - ee / 4.0 * y * y) * (1.0 - (1.0 - 2.0 * x) * t / QQ + (
                4.0 * x * (1.0 - x) + ee) / 4.0 / np.sqrt(1.0 + ee) * (t - tmin) / QQ)
                - (2.0 - y) ** 2 * (1.0 - x / 2.0 + (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) / 4.0 * (
                1.0 - t / QQ) + (4.0 * x * (1.0 - x) + ee) / 2.0 / np.sqrt(1.0 + ee) * (t - tmin) / QQ))
        # helicity - changing (F_eff)
        C_011 = 8.0 * np.sqrt(2.0) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (1.0 + ee) ** 2 * (
                (2.0 - y) ** 2 * (t - tmin) / QQ * (1.0 - x + ((1.0 - x) * x + ee / 4.0) / np.sqrt(1.0 + ee) * (t - tmin) / QQ)
                + (1.0 - y - ee / 4.0 * y * y) / np.sqrt(1 + ee) * (1.0 - (1.0 - 2.0 * x) * t / QQ) * (
                ee - 2.0 * (1.0 + ee / 2.0 / x) * x * t / QQ))
        C_011_V = 16.0 * np.sqrt(2.0) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * x * t / QQ * (
                (Ktilde_10 * (2.0 - y)) ** 2 / QQ + (1.0 - (1.0 - 2.0 * x) * t / QQ) ** 2 * (1.0 - y - ee / 4.0 * y * y))
        C_011_A = 8.0 * np.sqrt(2.0) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * t / QQ * (
                (Ktilde_10 * (2.0 - y)) ** 2 * (1.0 - 2.0 * x) / QQ + (1.0 - (1.0 - 2.0 * x) * t / QQ)
                * (1.0 - y - ee / 4.0 * y * y) * (4.0 - 2.0 * x + 3.0 * ee + t / QQ * (4.0 * x * (1.0 - x) + ee)))
        # n = 2 -----------------------------------------
        # helicity - conserving (F)
        C_112 = 8.0 * (2.0 - y) * (1.0 - y - ee / 4.0 * y * y) / (1.0 + ee) ** 2 * (
                2.0 * ee / np.sqrt(1.0 + ee) / (1.0 + np.sqrt(1.0 + ee)) * Ktilde_10 ** 2 / QQ + x * t * (
                t - tmin) / QQ / QQ * (1.0 - x - (np.sqrt(1.0 + ee) - 1.0) / 2.0 + ee / 2.0 / x))
        C_112_V = 8.0 * (2.0 - y) * (1.0 - y - ee / 4.0 * y * y) / (1.0 + ee) ** 2 * x * t / QQ * (
                4.0 * Ktilde_10 ** 2 / np.sqrt(1.0 + ee) / QQ + (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) / 2.0 * (1.0 + t / QQ) * (t - tmin) / QQ)
        C_112_A = 4.0 * (2.0 - y) * (1.0 - y - ee / 4.0 * y * y) / (1.0 + ee) ** 2 * t / QQ * (
                4.0 * (1.0 - 2.0 * x) * Ktilde_10 ** 2 / np.sqrt(1.0 + ee) / QQ - (3.0 - np.sqrt(1.0 + ee) - 2.0 * x + ee / x) * x * (t - tmin) / QQ)
        # helicity - changing (F_eff)
        C_012 = -8.0 * np.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * (1.0 + ee / 2.0) * (
                1.0 + (1.0 + ee / 2.0 / x) / (1.0 + ee / 2.0) * x * t / QQ)
        C_012_V = 8.0 * np.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * x * t / QQ * (
                1.0 - (1.0 - 2.0 * x) * t / QQ)
        C_012_A = 8.0 * np.sqrt(2.0) * K * (2.0 - y) * np.sqrt(1.0 - y - ee / 4.0 * y * y) / (1.0 + ee) ** 2 * t / QQ * (
                1.0 - x + (t - tmin) / 2.0 / QQ * (4.0 * x * (1.0 - x) + ee) / np.sqrt(1.0 + ee))
        # n = 3 -----------------------------------------
        # helicity - conserving (F)
        C_113 = -8.0 * K * (1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * (np.sqrt(1.0 + ee) - 1.0) * (
                (1.0 - x) * t / QQ + (np.sqrt(1.0 + ee) - 1.0) / 2.0 * (1.0 + t / QQ))
        C_113_V = -8.0 * K * (1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * x * t / QQ * (
                np.sqrt(1.0 + ee) - 1.0 + (1.0 + np.sqrt(1.0 + ee) - 2.0 * x) * t / QQ)
        C_113_A = 16.0 * K * (1.0 - y - ee / 4.0 * y * y) / (np.sqrt(1.0 + ee) ** 5) * t * (t - tmin) / QQ / QQ * (
                x * (1.0 - x) + ee / 4.0)

        # A_U_I, B_U_I and C_U_I
        A_U_I = C_110 + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_010 + (
                    C_111 + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_011) * np.cos(
            self.PI - (phi * self.RAD)) + (
                            C_112 + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_012) * np.cos(
            2.0 * (self.PI - (phi * self.RAD))) + C_113 * np.cos(3.0 * (self.PI - (phi * self.RAD)))
        B_U_I = xi / (1.0 + t / 2.0 / QQ) * (
                    C_110_V + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_010_V + (
                                C_111_V + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_011_V) * np.cos(
                self.PI - (phi * self.RAD)) + (
                                C_112_V + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * C_012_V) * np.cos(
                2.0 * (self.PI - (phi * self.RAD))) + C_113_V * np.cos(3.0 * (self.PI - (phi * self.RAD))))
        C_U_I = xi / (1.0 + t / 2.0 / QQ) * (
                    C_110 + C_110_A + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * (C_010 + C_010_A) + (
                                C_111 + C_111_A + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * (
                            C_011 + C_011_A)) * np.cos(self.PI - (phi * self.RAD)) + (
                                C_112 + C_112_A + np.sqrt(2.0) / (2.0 - x) * Ktilde_10 / np.sqrt(QQ) * f * (
                            C_012 + C_012_A)) * np.cos(2.0 * (self.PI - (phi * self.RAD))) + (C_113 + C_113_A) * np.cos(
                3.0 * (self.PI - (phi * self.RAD))))

        return A_U_I, B_U_I, C_U_I

    def curve_fit(self, kins, cffs):
        calc = F1F2()
        QQ, x, t, phi, k = kins[:, 0], kins[:, 1], kins[:, 2], kins[:, 3], kins[:, 4]
        F1, F2 = calc.f1_f21(t)  # calculating F1 and F2 using passed data as opposed to passing in F1 and F2
        ReH, ReE, ReHtilde, c0fit = cffs[:, 0], cffs[:, 1], cffs[:, 2], cffs[:, 3]  # output of network
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.set_kinematics(QQ, x, t, k)
        P1, P2 = self.bh_lepton_propagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.bh_uu(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.i_uu(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return f_pred


class TotalFLayer:
    def __init__(self):
        self.f = BHDVCStf()

    def call(self, inputs):
        return self.f.curve_fit(inputs[:, 0:5], inputs[:, 5:9])  # QQ, x, t, phi, k, cff1, cff2, cff3, cff4


def F2VsPhi(dataframe, SetNum, xdat, cffs):
    f = BHDVCStf().curve_fit
    TempFvalSilces = dataframe[dataframe["#Set"] == SetNum]
    TempFvals = TempFvalSilces["F"]
    TempFvals_sigma = TempFvalSilces["errF"]

    temp_phi = TempFvalSilces["phi_x"]
    plt.errorbar(temp_phi, TempFvals, TempFvals_sigma, fmt='.', color='blue', label="Data")
    plt.xlim(0, 368)
    temp_unit = (np.max(TempFvals) - np.min(TempFvals)) / len(TempFvals)
    plt.ylim(np.min(TempFvals) - temp_unit, np.max(TempFvals) + temp_unit)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc=4, fontsize=10, handlelength=3)
    plt.title("Local fit with data set #" + str(SetNum), fontsize=20)
    plt.plot(temp_phi, f(xdat, cffs), 'g--', label='fit')
    file_name = "plot_set_number_{}.png".format(SetNum)
    plt.savefig(file_name)


def cffs_from_globalModel(model, kinematics, numHL=1):
    '''
    :param model: the model from which the cffs should be predicted
    :param kinematics: the kinematics that should be used to predict
    :param numHL: the number of hidden layers:
    '''
    subModel = model.layers[numHL + 2].output
    return subModel(np.asarray(kinematics)[None, 0])[0]


class F1F2:
    def __init__(self):
        self.GM0 = 2.792847337

    def ffGE(self, t):
        GE = 1.0 / (1.0 + (-t / 0.710649)) ** 2
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
