import tensorflow as tf
import keras

class BKM10(keras.layers.Layer):

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

        # self.K_tilde = tf.sqrt((t_min-t)*((1.-x)*tf.sqrt(1.+e2) + (t_min-t)*(e2+4.*x*(1.-x))/(4.*Q2)))

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

    def calculate_cross_section(self, phi, cffs):
        """
        Computes the differential cross section for the BKM formalism in twist-2 for now,
        based on Eqn (1.1) BKM 2010 paper.

        Parameters:
            φ : azimuthal angle between leptonic and hadronic planes (radians)
            cffs ((sequence of float, shape (4,))) : 
                Compton form factors [Re(H), Re(H̃), Re(E), |Amplitude_DVCS|²].

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

        # ReH, ReHt, ReE, ReEt, ImH, ImHt, ImE, ImEt = tf.unstack(cffs, axis=1)
        ReH, ReHt, ReE, T_DVCS_2 = tf.unstack(cffs, axis=1)
        ImH = tf.constant(0.0, dtype=tf.float32)
        ImHt = tf.constant(0.0, dtype=tf.float32)
        ImE = tf.constant(0.0, dtype=tf.float32)

        T_BH_2 = self.compute_BH_amplitude(phi, P1, P2)

        I = self.compute_Interference(phi, P1, P2, ReH, ImH, ReHt, ImHt, ReE, ImE)

        # T_DVCS_2 = self.compute_DVCS_amplitude(phi, ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt)

        d_sigma = T_BH_2 + T_DVCS_2 + I
        
        return d_sigma

    def compute_BH_amplitude(self, phi, P1, P2):
        """
        Computes the Bethe-Heitler amplitude squared term in nb using inputs from compute_kinematics
        """

        c0_BH, c1_BH, c2_BH, s1_BH = self.BH_UP_coeffs()

        # BKM 2002 : Eq (25)
        TBH2 = 1./(self.x*self.x*self.y*self.y*(1.+self.e2)*(1.+self.e2)*self.t*P1*P2) * \
            (c0_BH + c1_BH*tf.cos(phi) + c2_BH *
             tf.cos(2.*phi) + s1_BH*tf.sin(phi))

        TBH2 *= self.GeV_to_nb*self.dsig_prefactor

        return TBH2

    def compute_DVCS_amplitude(self, phi, ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt):
        """
        Computes the DVCS amplitude squared term in nb using inputs from compute_kinematics
        """

        c0_DVCS, c1_DVCS, s1_DVCS = self.DVCS_UP_coeffs(
            ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt)

        # BKM 2010 : Eq (2.17)
        TDVCS2 = 1./(self.y*self.y*self.Q2) * (c0_DVCS + c1_DVCS *
                                               tf.cos(phi) + s1_DVCS*tf.sin(phi))

        TDVCS2 *= self.GeV_to_nb*self.dsig_prefactor

        return TDVCS2

    def compute_Interference(self, phi, P1, P2, ReH, ImH, ReHt, ImHt, ReE, ImE):
        """
        Computes the Interference term amplitude squared term in nb using inputs from compute_kinematics
        """

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
        """
        Computes the Fourier coefficients (c0, c1, c2, s1) of the unpolarized target Bethe-Heitler squared amplitude. 
        Based on Eq. (35-37) of the BKM 2002 paper.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            F1 (float): Dirac form factor
            F2 (float): Pauli form factor

        Returns:
            tuple: (c0, c1, c2, s1) Fourier coefficients
        """

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
        """
        Computes Fourier coefficients for the DVCS term in the unpolarized Bethe-Heitler + DVCS cross section, 
        based on Eqn (2.18, 2.19 and 2.22) BKM 2010 paper.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            F1 (float): Dirac form factor
            F2 (float): Pauli form factor
            ReH (float): Real part of H CFF
            ImH (float): Imaginary part of H CFF
            ReHt (float): Real part of H~ CFF
            ImHt (float): Imaginary part of H~ CFF
            ReE (float): Real part of E CFF
            ImE (float): Real part of E CFF

        Returns:
            tuple: (c0, c1, s1) 
            Returns total DVCS amplitude squared term Fourier coefficients
        """
        Q2, t, x, e2, y, K = self.Q2, self.t, self.x, self.e2, self.y, self.K

        @staticmethod
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
        Re_CC0A, Im_CC0A = CCDVCS_UP_of_F_Fstar(ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt,  # F
                                                ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt)  # F*

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
        Re_CC0B, Im_CC0B = CCDVCS_UP_of_F_Fstar(ReH_eff, ImH_eff, ReHt_eff, ImHt_eff, ReE_eff, ImE_eff, ReEt_eff, ImEt_eff,  # F
                                                ReH_eff, ImH_eff, ReHt_eff, ImHt_eff, ReE_eff, ImE_eff, ReEt_eff, ImEt_eff)  # F*

        c0_B = 16.*K*K*Re_CC0B/((2.-x)*(2.-x)*(1.+e2))

        Re_CC1, Im_CC1 = CCDVCS_UP_of_F_Fstar(ReH_eff, ImH_eff, ReHt_eff, ImHt_eff, ReE_eff, ImE_eff, ReEt_eff, ImEt_eff,  # F
                                              ReH, ImH, ReHt, ImHt, ReE, ImE, ReEt, ImEt)  # F*

        c1_A = 8.*K*(2.-y)*Re_CC1/((2.-x)*(1.+e2))

        s1_A = 8.*K*(-self.helc*y*tf.sqrt(1.+e2))*Im_CC1/((2.-x)*(1.+e2))

        c0 = c0_A + c0_B
        c1 = c1_A
        s1 = s1_A

        return c0, c1, s1

    def I_UP_coeffs(self, ReH, ImH, ReHt, ImHt, ReE, ImE):
        """
        Computes Fourier coefficients for the interference term in the unpolarized Bethe-Heitler + DVCS cross section, 
        based on Eqn (2.35) BKM 2010 paper.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            F1 (float): Dirac form factor
            F2 (float): Pauli form factor
            ReH (float): Real part of H CFF
            ImH (float): Imaginary part of H CFF
            ReHt (float): Real part of H~ CFF
            ImHt (float): Imaginary part of H~ CFF
            ReE (float): Real part of E CFF
            ImE (float): Real part of E CFF

        Returns:
            tuple: (c0, c1, c2, c3, s1, s2) 
            Returns total Interference term Fourier coefficients
        """

        # One would need 𝓕_eff and 𝓕_T to be able to include the helicity changing terms. Skip for now.

        @staticmethod
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

        # Second term of BKM 2010 : Eq (2.35) - Need to pass CFF_eff to coeffs_zero_plus_UP.
        c0_B, c1_B, c2_B, c3_B, s1_B, s2_B = self.coeffs_zero_plus_UP(
            ReCC_eff, ReCC_V_eff, ReCC_A_eff, ImCC_eff, ImCC_V_eff, ImCC_A_eff)

        # Third term of BKM 2010 : Eq (2.35) - Need to pass CFF_T to coeffs_minus_plus_UP. Skip for now.
        c0_C, c1_C, c2_C, c3_C, s1_C, s2_C = 0., 0., 0., 0., 0., 0.

        c0 = c0_A + c0_B + c0_C
        c1 = c1_A + c1_B + c1_C
        c2 = c2_A + c2_B + c2_C
        c3 = c3_A + c3_B + c3_C
        s1 = s1_A + s1_B + s1_C
        s2 = s2_A + s2_B + s2_C

        return c0, c1, c2, c3, s1, s2

    def coeffs_plus_plus_UP(self, ReCC, ReCC_V, ReCC_A, ImCC, ImCC_V, ImCC_A):
        """
        Computes helicity-conserving unpolarised target coefficients. Based on BKM 2010 - Eq (A1), Eq (2.35)
        These are all the plus-plus components.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            ReCC (float): Real CFF combinations, BKM 2010 : Eq (2.28)
            ReCC_V (float): Real CFF combinations, BKM 2010 : Eq (2.29)
            ReCC_A (float): Real CFF combinations, BKM 2010 : Eq (2.30)
            ImCC (float): Imaginary CFF combinations, BKM 2010 : Eq (2.28)
            ImCC_V (float): Imaginary CFF combinations, BKM 2010 : Eq (2.29)
            ImCC_A (float): Imaginary CFF combinations, BKM 2010 : Eq (2.30)

        Returns:
            tuple: (c0, c1, c2, c3, s1, s2)
            Returns plus-plus Fourier coefficients
        """

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

        # compute SS only when denominator is non-zero, else output ImCC

        SS1_pp = tf.where(tf.not_equal(s1_pp, 0.0),
                          ImCC + s1_pp_V*ImCC_V/s1_pp + s1_pp_A*ImCC_A/s1_pp,
                          ImCC)
        SS2_pp = tf.where(tf.not_equal(s2_pp, 0.0),
                          ImCC + s2_pp_V*ImCC_V/s2_pp + s2_pp_A*ImCC_A/s2_pp,
                          ImCC)

        # First term of BKM 2010 : Eq (2.35)

        c0 = c0_pp*CC0_pp
        c1 = c1_pp*CC1_pp
        c2 = c2_pp*CC2_pp
        c3 = c3_pp*CC3_pp
        s1 = s1_pp*SS1_pp
        s2 = s2_pp*SS2_pp

        return c0, c1, c2, c3, s1, s2

    def coeffs_zero_plus_UP(self, ReCC_eff, ReCC_V_eff, ReCC_A_eff, ImCC_eff, ImCC_V_eff, ImCC_A_eff):
        """
        Computes longitudinal-transverse unpolarised target helicity-changing coefficients. Based on BKM 2010 - Eq (A2-A3), Eq (2.35) 
        These are all the 0-plus components.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            ReCC_eff (float): Real CFF_effective combinations, BKM 2010 : Eq (2.28)
            ReCC_V_eff (float): Real CFF_effective combinations, BKM 2010 : Eq (2.29)
            ReCC_A_eff (float): Real CFF_effective combinations, BKM 2010 : Eq (2.30)
            ImCC_eff (float): Imaginary CFF_effective combinations, BKM 2010 : Eq (2.28)
            ImCC_V_eff (float): Imaginary CFF_effective combinations, BKM 2010 : Eq (2.29)
            ImCC_A_eff (float): Imaginary CFF_effective combinations, BKM 2010 : Eq (2.30)

        Returns:
            tuple: (c0, c1, c2, c3, s1, s2)
            Returns 0-plus Fourier coefficients
        """

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

        # compute SS only when denominator is non-zero, else output ImCC

        SS1_0p = tf.where(tf.not_equal(s1_0p, 0.0),
                          B*ImCC_eff + s1_0p_V*ImCC_V_eff/s1_0p + s1_0p_A*ImCC_A_eff/s1_0p,
                          B*ImCC_eff)
        SS2_0p = tf.where(tf.not_equal(s2_0p, 0.0),
                          B*ImCC_eff + s2_0p_V*ImCC_V_eff/s2_0p + s2_0p_A*ImCC_A_eff/s2_0p,
                          B*ImCC_eff)

        # Second term of BKM 2010 : Eq (2.35)

        c0 = c0_0p*CC0_0p
        c1 = c1_0p*CC1_0p
        c2 = c2_0p*CC2_0p
        c3 = 0.
        s1 = s1_0p*SS1_0p
        s2 = s2_0p*SS2_0p

        return c0, c1, c2, c3, s1, s2

    def coeffs_minus_plus_UP(self, ReCC_T, ReCC_V_T, ReCC_A_T, ImCC_T, ImCC_V_T, ImCC_A_T):
        """
        Computes transverse-transverse unpolarised target helicity-changing coefficients. Based on BKM 2010 - Eq (A4), Eq (2.35) 
        These are all the minus-plus components.

        Parameters:
            Q2 (float): Photon virtuality (GeV^2)
            t (float): Momentum transfer squared (GeV^2)
            x (float): Bjorken scaling variable
            e2 (float): ε², ratio of longitudinal to transverse photon flux, BKM 2010 - Eq (1.2)
            y (float): Lepton energy loss fraction
            K (float): Kinematic factor, BKM 2010 - below Eq (2.21)
            t_min (float): Minimum kinematically allowed value of t, BKM 2002 - Eq (31)
            K_tilde (float): Modified kinematic factor, BKM 2010 - Eq (2.5)
            helc (float): Lepton polarisation (can be 1.0, 0.0 or -1.0)
            ReCC_T (float): Real CFF_T combinations, BKM 2010 : Eq (2.28)
            ReCC_V_T (float): Real CFF_T combinations, BKM 2010 : Eq (2.29)
            ReCC_A_T (float): Real CFF_T combinations, BKM 2010 : Eq (2.30)
            ImCC_T (float): Imaginary CFF_T combinations, BKM 2010 : Eq (2.28)
            ImCC_V_T (float): Imaginary CFF_T combinations, BKM 2010 : Eq (2.29)
            ImCC_A_T (float): Imaginary CFF_T combinations, BKM 2010 : Eq (2.30)

        Returns:
            tuple: (c0, c1, c2, c3, s1, s2)
            Returns minus-plus Fourier coefficients
        """

        Q2, t, x, e2, y, K, t_min, K_tilde, helc = self.Q2, self.t, self.x, self.e2, self.y, self.K, self.t_min, self.K_tilde, self.helc

        EE = (1.+e2)*(1.+e2)
        RE = tf.sqrt(1.+e2)
        RE5 = tf.pow(RE, 5.)
        TQ = t/Q2
        KTQ = K_tilde*K_tilde/Q2
        A = (1.-y-e2*y*y/4.)
        tp = t-t_min
        TPQ = tp/Q2

        c0_mp = (8.*(2.-y)/(tf.pow(RE, 3.))) * \
            ((2.-y)*(2.-y)*(RE-1.)*KTQ*0.5/(1.+e2) +
             A*(1.-x-0.5*(RE-1.)+0.5*e2/x)*x*TQ*TPQ/RE)

        c1_mp = ((8.*K/tf.pow(RE, 3.)))*((2.-y)*(2.-y)*((2.-RE)/(1.+e2)) *
                                         ((RE-1.+e2)*(1.-TQ)*0.5/(2.-RE)-x*TQ) +
                                         (2.*A/RE)*((1.-RE+0.5*e2)*0.5/RE+TQ*(1.-3.*0.5*x+(x+0.5*e2)*0.5/RE)))

        c2_mp = (4.*(2.-y)*A*(1.+RE)/RE5) * \
            ((2.-3.*x)*TQ+(1.-2.*x+(2.*(1.-x))/(1.+RE)) *
             x*TQ/Q2 + (1.+(RE+x+(1.-x)*TQ)*TQ/(1.+RE))*e2)

        c3_mp = -(8.*K*A*(1.+RE+0.5*e2)/RE5) * \
            (1. + (1.+RE+0.5*e2/x)*x*TQ/(1.+RE+0.5*e2))

        s1_mp = helc*(4.*K*(2.-y)*y/EE) * \
            (1.-RE+2.*e2-2.*(1.+0.5*(RE-1.)/x)*x*TQ)

        s2_mp = helc*(2.*y*A*(1.+RE)/EE)*(e2-2.*(1.+0.5*e2/x)*x*TQ) * \
            (1.+(RE-1.+2.*x)*TQ/(1.+RE))

        c0_mp_V = (4.*(2.-y)*x*TQ/RE5)*(2.*KTQ*(2.-2.*y+y*y+0.5*y*y*e2)-(1.-(1.-2.*x)*TQ)*A) * \
            (RE-1.+(RE+1.-2.*x)*TQ)

        c1_mp_V = (8.*K*x*TQ/RE5)*(2.*(1.-(1.-2.*x)*TQ)*(2.-2.*y+y*y+0.5*y*y*e2) +
                                   A*(3.-RE-(3.*(1.-2.*x)+RE)*TQ))

        c2_mp_V = (4.*(2.-y)*A*x*TQ/RE5) * \
            (4.*KTQ+1.+RE+TQ*((1.-2.*x)*(1.-2.*x-RE)*TQ-2.+4.*x+2.*x*RE))

        c3_mp_V = (8.*K*A*x*TQ*(1.+RE)/RE5)*(1.-TQ*(1.-2.*x-RE)/(1.+RE))

        s1_mp_V = helc*(8.*K*y*(2.-y)*x*TQ*(1.+RE)/EE) * \
            (1.-TQ*(1.-2.*x-RE)/(1.+RE))

        s2_mp_V = helc*(4.*y*A*x*TQ*(1.+RE)/EE) * \
            (1.-(1.-2.*x)*TQ) * (1.-TQ*(1.-2.*x-RE)/(1.+RE))

        c0_mp_A = (4.*(2.-y)*TQ/EE)*(TPQ*A*(2.*x*x-e2-3.*x+x*RE) +
                                     (KTQ/RE)*(4.-2.*x*(2.-y)*(2.-y)-4.*y+y*y-tf.pow(RE, 3.)))

        c1_mp_A = (4.*K*TQ/RE5)*((2.-2.*y+y*y+0.5*y*y*e2)*(5.-4.*x+3.*e2-RE-TQ*(1.-e2-RE-2.*x*(4.-4.*x-RE))) +
                                 A*(8.+5.*e2-6.*x+2.*x*RE-TQ*(2.-e2+2.*RE-4.*x*(3.-3.*x+RE))))

        c2_mp_A = (16.*(2.-y)*A*TQ/tf.pow(RE, 3))*(KTQ*(1.-2.*x)/(1.+e2)-(1.-x)*(2.*x*x-e2-3.*x-x*RE)/(4.*x*(1.-x)+e2) -
                                                   TPQ*(2.*x*x-e2-3.*x-x*RE)*0.25/RE)

        c3_mp_A = (16.*K*A*TQ/EE)*(1.-x+TPQ*(4.*x*(1.-x)+e2)*0.25/RE)

        s1_mp_A = helc*(4.*K*y*(2.-y)*TQ/EE) * \
            (3.+2.*e2+RE-2.*x-2.*x*RE-TQ*(1.-2.*x)*(1.-2.*x-RE))

        s2_mp_A = helc*(2.*A*TQ/EE)*(4.-2.*x+3.*e2+TQ*(4.*x*(1.-x)+e2)) * \
            (1.+RE-TQ*(1.-2.*x-RE))

        # BKM 2010 : Eq (2.37) Maybe??

        B = tf.sqrt(2)*KTQ/(2.-x)

        CC0_mp = B*(ReCC_T + c0_mp_V*ReCC_V_T/c0_mp + c0_mp_A*ReCC_A_T/c0_mp)
        CC1_mp = B*(ReCC_T + c1_mp_V*ReCC_V_T/c1_mp + c1_mp_A*ReCC_A_T/c1_mp)
        CC2_mp = B*(ReCC_T + c2_mp_V*ReCC_V_T/c2_mp + c2_mp_A*ReCC_A_T/c2_mp)
        CC3_mp = B*(ReCC_T + c3_mp_V*ReCC_V_T/c3_mp + c3_mp_A*ReCC_A_T/c3_mp)

        # compute SS only when denominator is non-zero, else output ImCC

        SS1_mp = tf.where(tf.not_equal(s1_mp, 0.0),
                          B*ImCC_T + s1_mp_V*ImCC_V_T/s1_mp + s1_mp_A*ImCC_A_T/s1_mp,
                          B*ImCC_T)
        SS2_mp = tf.where(tf.not_equal(s2_mp, 0.0),
                          B*ImCC_T + s2_mp_V*ImCC_V_T/s2_mp + s2_mp_A*ImCC_A_T/s2_mp,
                          B*ImCC_T)

        # Last term of BKM 2010 : Eq (2.35)

        c0 = c0_mp*CC0_mp
        c1 = c1_mp*CC1_mp
        c2 = c2_mp*CC2_mp
        c3 = c3_mp*CC3_mp
        s1 = s1_mp*SS1_mp
        s2 = s2_mp*SS2_mp

        return c0, c1, c2, c3, s1, s2
