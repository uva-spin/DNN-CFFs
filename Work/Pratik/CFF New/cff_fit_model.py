import tensorflow as tf
import keras

class CFF_Fit_Model(keras.layers.Layer):

    def __init__(self, verbose=0, LR=1e-1, mod_LR_factor=0.8, mod_LR_patience=30, min_LR=1e-5, ES_patience=30):
        super().__init__()

        self.verbose = verbose
        self.lr = LR
        self.mod_LR_factor = mod_LR_factor
        self.mod_LR_patience = mod_LR_patience
        self.min_LR = min_LR
        self.ES_patience = ES_patience
        
    def create_model(self, layers, activation, initializer=None, summary=False):

        scaled_inputs = keras.Input(shape=(3,))

        hidden = keras.layers.Dense(layers[0], name = 'hidden_0',
                                    kernel_initializer=initializer, activation=activation)(scaled_inputs)

        for i in range(len(layers)-1):
            hidden = keras.layers.Dense(layers[i+1], name = f'hidden_{i+1}',
                                        kernel_initializer=initializer, activation=activation)(hidden)

        cff_123 = keras.layers.Dense(3, name='Re_CFFs')(hidden)
        DVCS = keras.layers.Dense(1, name='DVCS', activation='softplus')(hidden)

        predicted_cffs = keras.layers.Concatenate(name='predicted_CFFs')([cff_123, DVCS])

        self.model = keras.Model(inputs=scaled_inputs, outputs=predicted_cffs)
        
        if summary:
            print(self.model.summary())
    
    def fit_model(self, kinematics_class, scaled_inputs, outputs_tensor, epochs=500, batch=2):
        '''
        Parameters:
            model : created model
            kinematics_class : BKM10 class object
            scaled_inputs : scaled kinematics [Q², xB, t]
            outputs_tensor (sequence of float, shape(4,)) : output variables [φ, σ-exp, σ-err, weights] where
            - φ : angle (radians)
            - σ-exp : sampled d⁴ cross section for a replica (nb)
            - σ-err : experimental errors for the d⁴ cross sections
            - weights : normalised weights
        '''
        
        opt = keras.optimizers.Adam(self.lr)

        self.model.compile(optimizer=opt,
                      loss=self.compute_loss(kinematics_class, outputs_tensor))
        
        modifyLR = keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=self.mod_LR_factor, patience=self.mod_LR_patience, min_lr=self.min_LR, mode='auto')

        EarlyStop = keras.callbacks.EarlyStopping(
            monitor='loss', patience=self.ES_patience, restore_best_weights=True)
        
        history = self.model.fit(scaled_inputs, outputs_tensor,
                            epochs=epochs, batch_size=batch, 
                            callbacks=[modifyLR, EarlyStop], verbose=self.verbose)

        return history

    def compute_loss(self, kinematics_class, outputs_tensor):
        """
        Computes and returns the model loss.
        """
        def loss(y_true, y_pred):
            '''
            Parameters: 
                y_true (float): True outputs tensor [φ, σ-exp, σ-err, weights]
                y_pred (float): Model outputs, predicted CFFs tensor
            '''

            phi, dsig_true, errors, weight = tf.unstack(y_true, axis=1)

            # chi2, r2 = self.goodness_of_fit(kinematics_class, outputs_tensor, y_pred)

            dsig_pred = kinematics_class.calculate_cross_section(phi, y_pred)

            dsig_true = keras.ops.log10(dsig_true)
            dsig_pred = keras.ops.log10(dsig_pred)
    
            weighted_mae = tf.reduce_sum(weight * tf.abs(dsig_true - dsig_pred))

            return weighted_mae
        
        return loss
    
    def goodness_of_fit(self, kinematics_class, outputs_tensor, cffs):
        '''
        Computes X² and R² goodness of fit paramters. Can be used with any batch size, but techinally should be used when the CFFs have been predicted for all φ, i.e. for a batch size of 24
        '''
        phi_all, dsig_all_exp, dsig_all_err, dsig_weights = tf.unstack(outputs_tensor, axis=1)

        N = tf.shape(phi_all)[0]
        
        cffs_all = tf.tile(tf.expand_dims(cffs[0], axis=0), [N, 1])

        dsig_all_pred = kinematics_class.calculate_cross_section(phi_all, cffs_all)
        
        N = tf.cast(N, tf.float32)
        chi2 = tf.reduce_sum((dsig_all_exp-dsig_all_pred)*(dsig_all_exp-dsig_all_pred)/dsig_all_err)/(N-1.)

        ss_res = tf.reduce_sum((dsig_all_exp - dsig_all_pred)**2)
        ss_tot = tf.reduce_sum((dsig_all_exp - tf.reduce_mean(dsig_all_exp))**2)
        R2 = 1 - ss_res / ss_tot

        return chi2, R2