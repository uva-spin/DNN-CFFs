import tensorflow as tf
from BHDVCS_tf import BHDVCStf
from BHDVCS_tf import TotalFLayer

class Models:

    def tf_model1(self, data_length):
        initializer = tf.keras.initializers.HeNormal()
        #### QQ, x_b, t, phi, k ####
        inputs = tf.keras.Input(shape=(5))

        QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
        kinematics = tf.keras.layers.concatenate([QQ, x_b, t], axis=1)
        x1 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(kinematics)
        x2 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(x1)
        outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x2)
        #### QQ, x_b, t, phi, k, cffs ####
        total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)

        TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2

        tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name="tfmodel")

        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0085, data_length, 0.96, staircase=False, name=None
        )

        tfModel.compile(
            optimizer = tf.keras.optimizers.Adam(lr),
            loss = tf.keras.losses.MeanSquaredError()
        )

        return tfModel
    
class F_calc:
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
