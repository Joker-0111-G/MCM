import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 替换为你的图片路径
img1 = mpimg.imread('电站1_清洗价格影响趋势图.png')
img2 = mpimg.imread('电站2_清洗价格影响趋势图.png')
img3 = mpimg.imread('电站3_清洗价格影响趋势图.png')
img4 = mpimg.imread('电站4_清洗价格影响趋势图.png')

# 创建 2×2 子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 显示图片并关闭坐标轴
axs[0, 0].imshow(img1)
axs[0, 0].axis('off')
#axs[0, 0].set_title('图1')

axs[0, 1].imshow(img2)
axs[0, 1].axis('off')
#axs[0, 1].set_title('图2')

axs[1, 0].imshow(img3)
axs[1, 0].axis('off')
#axs[1, 0].set_title('图3')

axs[1, 1].imshow(img4)
axs[1, 1].axis('off')
#axs[1, 1].set_title('图4')

plt.tight_layout()
plt.savefig('拼图_2x2.png', dpi=300)
plt.show()
