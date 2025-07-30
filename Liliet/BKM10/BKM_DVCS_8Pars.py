import tensorflow as tf
import math
    
class BKM_DVCS(object):

    def __init__(self):
        self.ALP_INV = tf.constant(137.0359998)  # 1 / Electromagnetic Fine Structure Constant
        self.PI = tf.constant(3.1415926535)
        self.RAD = tf.constant(self.PI / 180.)
        self.M = tf.constant(0.938272)  # Mass of the proton in GeV
        self.GeV2nb = tf.constant(.389379 * 1000000)  # Conversion from GeV to NanoBar
        self.M2 = tf.constant(0.938272 * 0.938272)  # Mass of the proton  squared in GeV

    @tf.function
    def SetKinematics(self, QQ, x, t, k):
        ee = 4. * self.M2 * tf.pow(x, 2) / QQ  # epsilon squared
        y = tf.sqrt(QQ) / (tf.sqrt(ee) * k)  # lepton energy fraction
        xi = x * (1. + t / 2. / QQ) / (2. - x + x * t / QQ);  # Generalized Bjorken variable
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

        # BH polarized Fourier harmonics eqs. (35 - 37)
        # phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K = [tf.cast(i, np.float32) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K]]
        c0_BH = 8. * K * K * ((2. + 3. * ee) * (QQ / t) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * x * x * (F1 + F2) * (F1 + F2)) + (2. - y) * (2. - y) * ((
                2. + ee) * ((4. * x * x * self.M2 / t) * (1. + t / QQ) * (1. + t / QQ) + 4. * (1. - x) * (1. + x * t / QQ)) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 
                4. * x * x * (x + (1. - x + ee / 2.) * (1. - t / QQ) * (1. - t / QQ) - x * (1. - 2. * x) * t * t / (QQ * QQ)) * (F1 + F2) * (F1 + F2)) + 8. * (1. + ee) * (
                1. - y - ee * y * y / 4.) * (2. * ee * (1. - t / (4. * self.M2)) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) - x * x * (1. - t / QQ) * (1. - t / QQ) * (F1 + F2) * (F1 + F2))
        c1_BH = 8. * K * (2. - y) * ((4. * x * x * self.M2 / t - 2. * x - ee) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * x * x * (
                1. - (1. - 2. * x) * t / QQ) * (F1 + F2) * (F1 + F2))
        c2_BH = 8. * x * x * K * K * ((4. * self.M2 / t) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * (F1 + F2) * (F1 + F2))

        # BH squared amplitude eq (25) divided by e^6
        Amp2_BH = 1. / (x * x * y * y * (1. + ee) * (1. + ee) * t * P1 * P2) * (c0_BH + c1_BH * tf.cos(self.PI - (phi * self.RAD)) + c2_BH * tf.cos(2. * (self.PI - (phi * self.RAD))))
        Amp2_BH = self.GeV2nb * Amp2_BH  # convertion to nb

        return Gamma * Amp2_BH

    @tf.function
    def IUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10, f):
        # phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10 = [tf.cast(i, np.float32) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10]]
        # Get BH propagators and set the kinematics
        self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)

        # Get A_UU_I, B_UU_I and C_UU_I interference coefficients
        A_U_I, B_U_I, C_U_I = self.ABC_UU_I_10(phi, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10, f)

        # BH-DVCS interference squared amplitude
        I_10 = 1. / (x * y * y * y * t * P1 * P2) * (A_U_I * (F1 * ReH - t / 4. / self.M2 * F2 * ReE) + B_U_I * (F1 + F2) * (ReH + ReE) + C_U_I * (F1 + F2) * ReHtilde)
        I_10 = self.GeV2nb * I_10  # convertion to nb

        return Gamma * I_10

    @tf.function
    def ABC_UU_I_10(self, phi, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10, f):  # Get A_UU_I, B_UU_I and C_UU_I interference coefficients BKM10

        # Interference coefficients  (BKM10 Appendix A.1)
        # n = 0 -----------------------------------------
        # helicity - conserving (F)
        C_110 = - 4. * (2. - y) * (1. + tf.sqrt(1 + ee)) / tf.pow((1. + ee), 2) * (Ktilde_10 * Ktilde_10 * (2. - y) * (2. - y) / QQ / tf.sqrt(1 + ee)
                + t / QQ * (1. - y - ee / 4. * y * y) * (2. - x) * (1. + (2. * x * (2. - x + (tf.sqrt(1. + ee) - 1.) / 2. + ee / 2. / x) * t / QQ + ee) / (2. - x) / (1. + tf.sqrt(1. + ee))))
        C_110_V = 8. * (2. - y) / tf.pow((1. + ee), 2) * x * t / QQ * ((2. - y) * (2. - y) / tf.sqrt(1. + ee) * Ktilde_10 * Ktilde_10 / QQ
                + (1. - y - ee / 4. * y * y) * (1. + tf.sqrt(1. + ee)) / 2. * (1. + t / QQ) * (1. + (tf.sqrt(1. + ee) - 1. + 2. * x) / (1. + tf.sqrt(1. + ee)) * t / QQ))
        C_110_A = 8. * (2. - y) / tf.pow((1. + ee), 2) * t / QQ * ((2. - y) * (2. - y) / tf.sqrt(1. + ee) * Ktilde_10 * Ktilde_10 / QQ * (
                1. + tf.sqrt(1. + ee) - 2. * x) / 2.+ (1. - y - ee / 4. * y * y) * ((1. + tf.sqrt(1. + ee)) / 2. * (1. + tf.sqrt(1. + ee) - x + (
                tf.sqrt(1. + ee) - 1. + x * (3. + tf.sqrt(1. + ee) - 2. * x) / (1. + tf.sqrt(1. + ee)))* t / QQ) - 2. * Ktilde_10 * Ktilde_10 / QQ))
        # helicity - changing (F_eff)
        C_010 = 12. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * (ee + (2. - 6. * x - ee) / 3. * t / QQ)
        C_010_V = 24. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),5) * x * t / QQ * (1. - (1. - 2. * x) * t / QQ)
        C_010_A = 4. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),5) * t / QQ * (8. - 6. * x + 5. * ee) * (
                1. - t / QQ * ((2. - 12 * x * (1. - x) - ee)/ (8. - 6. * x + 5. * ee)))
        # n = 1 -----------------------------------------
        # helicity - conserving (F)
        C_111 = -16. * K * (1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * ((1. + (1. - x) * (tf.sqrt(1 + ee) - 1.) / 2. / x + ee / 4. / x) * x * t / QQ - 3. * ee / 4.) - 4. * K * (
                2. - 2. * y + y * y + ee / 2. * y * y) * (1. + tf.sqrt(1 + ee) - ee) / tf.pow(tf.sqrt(1. + ee), 5) * (1. - (1. - 3. * x) * t / QQ + (
                1. - tf.sqrt(1 + ee) + 3. * ee) / (1. + tf.sqrt(1 + ee) - ee) * x * t / QQ)
        C_111_V = 16. * K / tf.pow(tf.sqrt(1. + ee), 5) * x * t / QQ * ((2. - y) * (2. - y) * (1. - (1. - 2. * x) * t / QQ) + (
                1. - y - ee / 4. * y * y)* (1. + tf.sqrt(1. + ee) - 2. * x) / 2. * (t - tmin) / QQ)
        C_111_A = -16. * K / tf.pow((1. + ee), 2) * t / QQ * ((1. - y - ee / 4. * y * y) * (1. - (1. - 2. * x) * t / QQ + (
                4. * x * (1. - x) + ee) / 4. / tf.sqrt(1. + ee) * (t - tmin) / QQ)- tf.pow((2. - y), 2) * (1. - x / 2. + (1. + tf.sqrt(1. + ee) - 2. * x) / 4. * (
                1. - t / QQ) + (4. * x * (1. - x) + ee) / 2. / tf.sqrt(1. + ee) * (t - tmin) / QQ))
        # helicity - changing (F_eff)
        C_011 = 8. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow((1. + ee), 2) * (tf.pow((2. - y), 2) * (t - tmin) / QQ * (
                1. - x + ((1. - x) * x + ee / 4.) / tf.sqrt(1. + ee) * (t - tmin) / QQ) + (1. - y - ee / 4. * y * y) / tf.sqrt(1 + ee) * (
                1. - (1. - 2. * x) * t / QQ) * (ee - 2. * (1. + ee / 2. / x) * x * t / QQ))
        C_011_V = 16. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * x * t / QQ * (
                tf.pow(Ktilde_10 * (2. - y), 2) / QQ + tf.pow((1. - (1. - 2. * x) * t / QQ), 2) * (1. - y - ee / 4. * y * y))
        C_011_A = 8. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * t / QQ * (tf.pow(Ktilde_10 * (2. - y), 2) * (1. - 2. * x) / QQ + (
                1. - (1. - 2. * x) * t / QQ) * (1. - y - ee / 4. * y * y) * (4. - 2. * x + 3. * ee + t / QQ * (4. * x * (1. - x) + ee)))
        # n = 2 -----------------------------------------
        # helicity - conserving (F)
        C_112 = 8. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),2) * (2. * ee / tf.sqrt(1. + ee) / (1. + tf.sqrt(1. + ee)) * tf.pow(
                Ktilde_10, 2) / QQ + x * t * (t - tmin) / QQ / QQ * (1. - x - (tf.sqrt(1. + ee) - 1.) / 2. + ee / 2. / x))
        C_112_V = 8. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),2) * x * t / QQ * (
                4. * tf.pow(Ktilde_10, 2) / tf.sqrt(1. + ee) / QQ + (1. + tf.sqrt(1. + ee) - 2. * x) / 2. * (1. + t / QQ) * (t - tmin) / QQ)
        C_112_A = 4. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),2) * t / QQ * (4. * (1. - 2. * x) * 
                tf.pow(Ktilde_10, 2) / tf.sqrt(1. + ee) / QQ - (3. - tf.sqrt(1. + ee) - 2. * x + ee / x) * x * (t - tmin) / QQ)
        # helicity - changing (F_eff)
        C_012 = -8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * (1. + ee / 2.) * (1. + (1. + ee / 2. / x) / (1. + ee / 2.) * x * t / QQ)
        C_012_V = 8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),5) * x * t / QQ * (1. - (1. - 2. * x) * t / QQ)
        C_012_A = 8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow((1. + ee), 2) * t / QQ * (1. - x + (t - tmin) / 2. / QQ * (4. * x * (1. - x) + ee) / tf.sqrt(1. + ee))
        # n = 3 -----------------------------------------
        # helicity - conserving (F)
        C_113 = -8. * K * (1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),5) * (tf.sqrt(1. + ee) - 1.) * ((1. - x) * t / QQ + (tf.sqrt(1. + ee) - 1.) / 2. * (1. + t / QQ))
        C_113_V = -8. * K * (1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * x * t / QQ * (tf.sqrt(1. + ee) - 1. + (1. + tf.sqrt(1. + ee) - 2. * x) * t / QQ)
        C_113_A = 16. * K * (1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * t * (t - tmin) / QQ / QQ * (x * (1. - x) + ee / 4.)

        # A_U_I, B_U_I and C_U_I
        A_U_I = C_110 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(QQ) * f * C_010 + (C_111 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * C_011) * tf.cos(self.PI - (phi * self.RAD)) + (C_112 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(QQ) * f * C_012) * tf.cos(
                2. * (self.PI - (phi * self.RAD))) + C_113 * tf.cos(3. * (self.PI - (phi * self.RAD)))
        B_U_I = xi / (1. + t / 2. / QQ) * (C_110_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(QQ) * f * C_010_V + (C_111_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * C_011_V) * tf.cos(self.PI - (phi * self.RAD)) + (C_112_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * C_012_V) * tf.cos(2. * (self.PI - (phi * self.RAD))) + C_113_V * tf.cos(3. * (self.PI - (phi * self.RAD))))
        C_U_I = xi / (1. + t / 2. / QQ) * (C_110 + C_110_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(QQ) * f * (C_010 + C_010_A) + (C_111 + C_111_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / 
                tf.sqrt(QQ) * f * (C_011 + C_011_A)) * tf.cos(self.PI - (phi * self.RAD)) + (C_112 + C_112_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * (C_012 + C_012_A)) * tf.cos(2. * (self.PI - (phi * self.RAD))) + (C_113 + C_113_A) * tf.cos(3. * (self.PI - (phi * self.RAD))))

        return A_U_I, B_U_I, C_U_I

    @tf.function
    def DVCS(self, QQ, x, t, phi, ee, y, K, Gamma, ReH, ReE, ReHt, ReEt, ImH, ImE, ImHt, ImEt, f):

        H = tf.complex(ReH, ImH)
        E = tf.complex(ReE, ImE)
        Ht = tf.complex(ReHt, ImHt)
        Et = tf.complex(ReEt, ImEt)

        #  c_dvcs_unp(F,F*) coefficients (BKM10 eqs. [2.22]) for pure DVCS
        c_dvcs_ffs = tf.cast(QQ * ( QQ + x * t ) / tf.pow( ( ( 2. - x ) * QQ + x * t ), 2),tf.complex64) * (  tf.cast(4. * ( 1. - x ), tf.complex64) * (H*tf.math.conj(H)) +  tf.cast(4. * ( 1. - x + ( 2. * QQ + t ) / ( QQ + x * t ) * ee / 4. ), tf.complex64) *  (Ht*tf.math.conj(Ht))
                    - tf.cast(x * x * tf.pow( ( QQ + t ), 2) / ( QQ * ( QQ + x * t ) ), tf.complex64) * ( H*tf.math.conj(E) + E*tf.math.conj(H) ) - tf.cast(x * x * QQ / ( QQ + x * t ), tf.complex64) * ( Ht*tf.math.conj(Et) + Et*tf.math.conj(Ht) )
                    - tf.cast(( x * x * tf.pow( ( QQ + t ), 2) / ( QQ * ( QQ + x * t ) ) + tf.pow( ( ( 2. - x ) * QQ + x * t ), 2) / QQ / ( QQ + x * t ) * t / 4. / self.M2 ), tf.complex64) * (E*tf.math.conj(E))
                    - tf.cast(( x * x * QQ / ( QQ + x * t ) * t / 4. / self.M2 ), tf.complex64) * (Et*tf.math.conj(Et)) )
        c_dvcs_ffs = tf.cast(c_dvcs_ffs, tf.float32)
        
        # c_dvcs_unp(Feff,Feff*)
        c_dvcs_effeffs = f * f * c_dvcs_ffs
        # c_dvcs_unp(Feff,F*)
        c_dvcs_efffs = f * c_dvcs_ffs

        #  dvcs c_n coefficients (BKM10 eqs. [2.18], [2.19])
        c0_dvcs_10 = 2. * ( 2. - 2. * y + y * y  + ee / 2. * y * y ) / ( 1. + ee ) * c_dvcs_ffs + 16. * K * K / tf.pow(( 2. - x ), 2) / ( 1. + ee ) * c_dvcs_effeffs
        c1_dvcs_10 = 8. * K / ( 2. - x ) / ( 1. + ee ) * ( 2. - y ) * c_dvcs_efffs

        Amp2_DVCS_10 = 1. / ( y * y * QQ ) * ( c0_dvcs_10 + c1_dvcs_10 * (tf.cos(self.PI - (phi * self.RAD))) )

        Amp2_DVCS_10 = self.GeV2nb * Amp2_DVCS_10; # convertion to nb

        return  Gamma * Amp2_DVCS_10

    @tf.function
    def total_xs(self, kins, cffs, twist):

        # Split input data
        k, QQ, x, t, phi = tf.split(kins, num_or_size_splits=5, axis=1)
        ReH, ReE, ReHt, ReEt, ImH, ImE, ImHt, ImEt = tf.split(cffs, num_or_size_splits=8, axis=1) 

        # Compute kinematic dependent variables
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.SetKinematics(QQ, x, t, k)

        # Twist-2 approximation selection
        if twist == "t2": # F_eff = 0 ( pure twist 2) --> DVCS is constant
            f = 0  
        if twist == "t2_ho": # twist-2 higher order (ho) corrections from t3 (DVCS phi-dependence appears)
            f = - 2. * xi / (1. + xi)
        if twist == "t2_ww": # twist-2 higher order (ho) corrections from t3 using the WW relations (DVCS phi-dependence appears)
            f = 2. / (1. + xi)

        # Get elastic FFs
        ffs = FFs()
        F1, F2 = ffs.F1_F2(t) 
        # Get lepton propagators
        P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)

        xsbhuu = self.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K) # BH cross-section
        xsiuu = self.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHt, tmin, xi, Ktilde_10, f) # BH-DVCS interference 
        xsdvcs = self.DVCS(QQ, x, t, phi, ee, y, K, Gamma, ReH, ReE, ReHt, ReEt, ImH, ImE, ImHt, ImEt, f)   # Pure DVCS cross-section

        f_pred = xsbhuu + xsiuu + xsdvcs # Total DVCS cross-section

        return f_pred
    
    @tf.function
    def total_xs_fix_dvcs(self, kins, pars,twist):

        # Split input data
        k, QQ, x, t, phi = tf.split(kins, num_or_size_splits=5, axis=1)
        ReH, ReE, ReHt, dvcs = tf.split(pars, num_or_size_splits=4, axis=1) 

        # Compute kinematic dependent variables
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.SetKinematics(QQ, x, t, k)

        # Twist-2 approximation selection
        if twist == "t2": # F_eff = 0 ( pure twist 2) --> DVCS is constant
            f = 0  
        if twist == "t2_ho": # twist-2 higher order (ho) corrections from t3 (DVCS phi-dependence appears)
            f = - 2. * xi / (1. + xi)
        if twist == "t2_ww": # twist-2 higher order (ho) corrections from t3 using the WW relations (DVCS phi-dependence appears)
            f = 2. / (1. + xi)

        # Get elastic FFs
        ffs = FFs()
        F1, F2 = ffs.F1_F2(t) 
        # Get lepton propagators
        P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)

        xsbhuu = self.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K) # BH cross-section
        xsiuu = self.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHt, tmin, xi, Ktilde_10, f) # BH-DVCS interference
        f_pred = xsbhuu + xsiuu + dvcs # Total DVCS cross-section

        return f_pred


class FFs:

    GM0 = 2.792847337
    M = 0.938272
    # Kelly's parametrization fit Parameters
    a1_GEp = -0.24
    b1_GEp = 10.98
    b2_GEp = 12.82
    b3_GEp = 21.97
    a1_GMp = 0.12
    b1_GMp = 10.97
    b2_GMp = 18.86
    b3_GMp = 6.55
     
    def tau(self, t):       
        tau = - t / 4. / FFs.M / FFs.M
        return tau

    def GEp(self, t):
        GEp = ( 1. + FFs.a1_GEp * self.tau(t) )/( 1. + FFs.b1_GEp * self.tau(t) + FFs.b2_GEp * self.tau(t) * self.tau(t) + FFs.b3_GEp * self.tau(t) * self.tau(t) * self.tau(t) )
        return GEp

    def GMp(self, t):
        GMp = FFs.GM0 * ( 1. + FFs.a1_GMp * self.tau(t) )/( 1. + FFs.b1_GMp * self.tau(t) + FFs.b2_GMp * self.tau(t) * self.tau(t) + FFs.b3_GMp * self.tau(t) * self.tau(t) * self.tau(t) )
        return GMp

    def F2(self, t):
        f2 = ( self.GMp(t) - self.GEp(t) ) / ( 1. + self.tau(t) )
        return f2

    def F1(self, t):
        f1 = ( self.GMp(t) - self.F2(t) )
        return f1

    def F1_F2(self, t):
        return self.F1(t), self.F2(t)
    

bkm = BKM_DVCS()
xs = bkm.total_xs_fix_dvcs(tf.constant([[5.75,1.82,0.343,-0.172, 7.5]]), tf.constant([[ -2.56464,2.21195,1.39564, 0.0315875]]), "t2")
print(xs)
