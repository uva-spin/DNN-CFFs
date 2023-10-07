import tensorflow as tf
from BHDVCS_tf import TotalFLayer
from BHDVCS_tf import BHDVCStf

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
  
class Models:
  def __init__(self):
    pass

  def tf_model1(self, data_length):
    initializer = tf.keras.initializers.HeNormal()

    kinematics = tf.keras.Input(shape=(3))
    x1 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(kinematics)
    x2 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(x1)
    outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x2)
    noncffInputs = tf.keras.Input(shape=(7))
    #### phi, kin1, kin2, kin3, kin4, F1, F2 ####
    total_FInputs = tf.keras.layers.concatenate([noncffInputs,outputs], axis=1)
    TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2

    tfModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs = TotalF, name="tfmodel")

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0085, data_length/1, 0.96, staircase=False, name=None
    )

    tfModel.compile(
        optimizer = tf.keras.optimizers.Adam(lr),
        loss = tf.keras.losses.MeanSquaredError()
    )

    return tfModel