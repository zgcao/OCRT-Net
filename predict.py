"""
OCRT-Net (Optical Component Reconstruction and Transfer Network)
Multi-Sensor Inference Wrapper for Community Use (OLCI / VIIRS / MSI)
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sensor = 'OLCI'

class OCRT_Predictor:
    SUPPORTED_SENSORS = {
        'OLCI':  ['OLCI_413', 'OLCI_443', 'OLCI_490', 'OLCI_510', 'OLCI_560', 'OLCI_620', 
                 'OLCI_665', 'OLCI_674', 'OLCI_681', 'OLCI_709', 'OLCI_754', 'OLCI_779'],
        'VIIRS': ['VIIRS_412', 'VIIRS_443', 'VIIRS_488', 'VIIRS_551', 'VIIRS_671', 'VIIRS_746'],
        'MSI':   ['MSI_443', 'MSI_490', 'MSI_560', 'MSI_665', 'MSI_705', 'MSI_740', 'MSI_783']
    }
    def __init__(self, sensor='OLCI', weights_path=None, base_dir='.'):
        """
        初始化多传感器 OCRT-Net 推理引擎
        :param sensor: 'OLCI', 'VIIRS', 或 'MSI'
        :param weights_path: 权重路径。若不填，将自动寻找 release/OCRT_Net_Production_{sensor}_Weights.h5
        :param base_dir: 存放物理基底 (.npz) 文件的根目录
        """
        self.sensor = sensor.upper()
        if self.sensor not in self.SUPPORTED_SENSORS:
            raise ValueError(f"不支持的传感器类型: {self.sensor}。支持的列表: {list(self.SUPPORTED_SENSORS.keys())}")
            
        self.expected_bands = self.SUPPORTED_SENSORS[self.sensor]
        self.base_dir = base_dir
        
        # 自动路由权重路径
        if weights_path is None:
            self.weights_path = os.path.join(base_dir, f'release/OCRT_Net_Production_{self.sensor}.weights.h5')
        else:
            self.weights_path = weights_path
            
        print(f"正在初始化 OCRT-Net [{self.sensor}] 推理引擎...")
        
        self._load_physical_bases()
        self.model = self._build_infer_model()
        
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"找不到专属权重文件: {self.weights_path}")
        self.model.load_weights(self.weights_path)
        print(f"[{self.sensor}] 预训练权重及物理基底加载成功！就绪。\n")

    def _load_physical_bases(self):
        """动态加载对应传感器的物理基底"""
        try:
            pca_path = os.path.join(self.base_dir, f'data/aph_pca_{self.sensor}.npz')
            water_path = os.path.join(self.base_dir, f'data/water_iops_{self.sensor}.npz')
            
            pca_data = np.load(pca_path)
            self.TF_APHI_MEAN = tf.constant(pca_data['mean'], dtype=tf.float32)
            self.TF_APHI_PC1  = tf.constant(pca_data['pc1'], dtype=tf.float32)
            self.TF_APHI_PC2  = tf.constant(pca_data['pc2'], dtype=tf.float32)

            water_data = np.load(water_path)
            self.TF_AW  = tf.constant(water_data['aw'], dtype=tf.float32)
            self.TF_BBW = tf.constant(water_data['bbw'], dtype=tf.float32)
            self.WAVELENGTHS = water_data['wavelengths']
            self.TF_WAVELENGTHS = tf.constant(self.WAVELENGTHS, dtype=tf.float32)
            self.NUM_BANDS = len(self.WAVELENGTHS)
            
            if self.NUM_BANDS != len(self.expected_bands):
                raise ValueError(f"物理基底波段数({self.NUM_BANDS})与预期特征数({len(self.expected_bands)})不匹配！")
                
        except Exception as e:
            raise RuntimeError(f"物理基底加载失败，请确保 {pca_path} 和 {water_path} 存在。详细错误: {e}")


    def _build_infer_model(self):
        """内部方法：构建包含物理嵌入层的网络拓扑"""
        predictor = self
        
        # 定义自定义物理层
        class AdgLayer(layers.Layer):
            def call(self, inputs):
                amp, slope = inputs[:, 0:1], inputs[:, 1:2]
                s_constrained = tf.nn.sigmoid(slope) * 0.014 + 0.008 
                # 这里用 predictor 替代原来的 self
                lam = tf.expand_dims(predictor.TF_WAVELENGTHS, axis=0)
                return amp * tf.exp(-s_constrained * (lam - 443.0))

        class BbpLayer(layers.Layer):
            def call(self, inputs):
                amp, eta = inputs[:, 0:1], inputs[:, 1:2]
                eta_constrained = tf.nn.sigmoid(eta) * 2.5
                lam = tf.expand_dims(predictor.TF_WAVELENGTHS, axis=0)
                return amp * tf.pow((lam / 443.0), -eta_constrained)

        class AphPcaLayer(layers.Layer):
            def call(self, inputs):
                amp, w1, w2 = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
                mean = tf.expand_dims(predictor.TF_APHI_MEAN, axis=0)
                pc1 = tf.expand_dims(predictor.TF_APHI_PC1, axis=0)
                pc2 = tf.expand_dims(predictor.TF_APHI_PC2, axis=0)
                return amp * (mean + w1 * pc1 + w2 * pc2)

        class GordonRTM(layers.Layer):
            def call(self, inputs):
                a_ph, a_dg, b_bp = inputs
                a_total = predictor.TF_AW + a_ph + a_dg
                bb_total = predictor.TF_BBW + b_bp
                u = bb_total / (a_total + bb_total + 1e-5)
                r_rs = 0.089 * u + 0.125 * tf.square(u)
                return (0.52 * r_rs) / (1.0 - 1.7 * r_rs + 1e-7)

        # 构建网络结构
        inputs_rrs = layers.Input(shape=(self.NUM_BANDS,), name='input_rrs')
        x = layers.Dense(128, activation='relu')(inputs_rrs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        
        adg_params = layers.Dense(2, activation='linear')(x)
        adg_merged = layers.Concatenate()([layers.Activation('softplus')(adg_params[:, 0:1]), adg_params[:, 1:2]])
        adg_rec = AdgLayer()(adg_merged)
        
        bbp_params = layers.Dense(2, activation='linear')(x)
        bbp_merged = layers.Concatenate()([layers.Activation('softplus')(bbp_params[:, 0:1]), bbp_params[:, 1:2]])
        bbp_rec = BbpLayer()(bbp_merged)
        
        aph_params = layers.Dense(3, activation='linear')(x)
        aph_merged = layers.Concatenate()([layers.Activation('softplus')(aph_params[:, 0:1]), aph_params[:, 1:]])
        aph_rec = AphPcaLayer()(aph_merged)
        
        rrs_rec = GordonRTM(name='rrs_rec')([aph_rec, adg_rec, bbp_rec])
        
        l2_reg = l2(1e-3)
        chla_hidden = layers.Dense(16, activation='elu', kernel_regularizer=l2_reg)(aph_merged)
        chla_pred = layers.Dense(1, activation='linear', name='chla_pred')(chla_hidden)
        
        spm_hidden = layers.Dense(16, activation='elu', kernel_regularizer=l2_reg)(bbp_merged)
        spm_pred = layers.Dense(1, activation='linear', name='spm_pred')(spm_hidden)
        
        ag_features = layers.Concatenate()([adg_merged, bbp_merged])
        ag_hidden = layers.Dense(16, activation='elu', kernel_regularizer=l2_reg)(ag_features)
        ag_pred = layers.Dense(1, activation='linear', name='ag_pred')(ag_hidden)
        
        # 只返回推断模型
        infer_model = Model(inputs=inputs_rrs, outputs=[rrs_rec, chla_pred, spm_pred, ag_pred, aph_rec, adg_rec, bbp_rec])
        return infer_model

    def predict(self, rrs_matrix, batch_size=64):
        if rrs_matrix.shape[1] != self.NUM_BANDS:
            raise ValueError(f"输入维度错误！[{self.sensor}] 模型需要 {self.NUM_BANDS} 个波段。")
        preds = self.model.predict(rrs_matrix, batch_size=batch_size, verbose=0)
        return {
            'Rrs_reconstructed': preds[0],
            'Chla_predicted': 10 ** preds[1].flatten(),
            'SPM_predicted':  10 ** preds[2].flatten(),
            'ag440_predicted': 10 ** preds[3].flatten()
        }

    def img_est_ocrt(self, data,batch_size=8192):
        """
        执行整景影像的掩膜提取、批量推断与影像重构
        """
        expected_features = model.NUM_BANDS
        if data.shape[-1] != expected_features:
            raise ValueError(f"输入特征维度错误！OCRT-Net 需要 {expected_features} 个波段，但传入了 {data.shape[-1]} 个。")
        im_shape = data.shape[:-1]
        # 1. 展平图像 (H*W, Bands) 并创建 Mask
        im_data_flat = data.reshape((-1, expected_features))
        # 寻找无效像素 (NaN 或 小于等于 0)
        # 因为我们在读取时给无法 log 插值的 779 赋了 NaN，这里会自动将它们判定为 invalid 并剔除
        invalid_mask = np.any(np.isnan(im_data_flat) | (im_data_flat <= 0), axis=1)
        # 提取纯净的有效像素矩阵送入网络
        valid_rrs = im_data_flat[~invalid_mask]
        print(f"      -> 影像有效水体像素数量: {len(valid_rrs)} / {len(im_data_flat)}")
        # 2. 初始化结果数组 (全部填充满 NaN)
        chl_flat = np.full(im_data_flat.shape[0], np.nan)
        spm_flat = np.full(im_data_flat.shape[0], np.nan)
        ag443_flat = np.full(im_data_flat.shape[0], np.nan)
        # 3. 如果有效像素不为 0，则调用 OCRT-Net 进行批量推断
        if len(valid_rrs) > 0:
            # OCRT_Predictor 内部处理推理
            results = self.model.predict(valid_rrs, batch_size=batch_size)
            # 物理截断 (剔除极端的物理外推值)
            chla_preds = results['Chla_predicted']
            spm_preds  = results['SPM_predicted']
            ag_preds   = results['ag440_predicted']
            chla_preds = np.where((chla_preds < 0.01) | (chla_preds > 500), np.nan, chla_preds)
            spm_preds  = np.where((spm_preds < 0.1)   | (spm_preds > 500), np.nan, spm_preds)
            # 将预测结果填回对应的有效索引位置
            chl_flat[~invalid_mask] = chla_preds
            spm_flat[~invalid_mask] = spm_preds
            ag443_flat[~invalid_mask] = ag_preds
        # 4. 重新折叠回二维影像形状 (H, W)
        chl_img = chl_flat.reshape(im_shape)
        spm_img = spm_flat.reshape(im_shape)
        ag443_img = ag443_flat.reshape(im_shape)
        return chl_img, spm_img, ag443_img
