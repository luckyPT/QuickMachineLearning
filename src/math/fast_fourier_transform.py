import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def freq_amplitude_phase():
    """参考：https://www.kancloud.cn/wizardforcel/hyry-studio-scipy/129098"""
    sampling_rate = 8000  # 采样频率，最好是分量信号频率的2倍
    fft_size = 512  # 采样点数量 至少
    t = np.arange(0, 1.0, 1.0 / sampling_rate)
    x = np.sin(2 * np.pi * 156.25 * t + 1) + 2 * np.sin(2 * np.pi * 234.375 * t + 2) + 3 * np.sin(
        2 * np.pi * 625 * t + 3)
    xs = x[:fft_size]
    # rfft 返回fft_size/2+1个复数，表示范围为：0(Hz)到sampling_rate/2(Hz)
    xf = np.fft.rfft(xs) / fft_size  # 变换之后的数据进行归一化
    freqs = np.linspace(0, sampling_rate / 2, fft_size // 2 + 1)  # 频率
    xfp = 2 * np.abs(xf)  # 复数模长表示振幅的一半
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t[:fft_size], xs, label="时域信号")
    plt.legend()
    plt.xlabel(u"时间(秒)")
    plt.title(u"156.25Hz和234.375Hz的波形和频谱")
    plt.subplot(212)
    plt.plot(freqs, xfp, label='振幅')
    plt.xlabel(u"频率(Hz)")
    angle = np.angle(xf) + np.pi / 2  # 相位角
    plt.plot(freqs, angle, label='相位')
    plt.subplots_adjust(hspace=0.4)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    freq_amplitude_phase()
