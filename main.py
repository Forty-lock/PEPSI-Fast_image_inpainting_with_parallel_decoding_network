import tensorflow as tf
import scipy.misc as sci
import os
import numpy as np
import Read_Image_List as ri
import module as mm
import ops as op
import random
import time
import cv2

Height = 256
Width = 256
batch_size = 8
mask_size = 128

dPath_l = ('./List')

dPath_train = ('/train_fh256.txt')
dPath_test = ('/test_fh256.txt')
dPath_testm = ('/test_mask256.txt')
dPath_testf = ('/test_maskff.txt')

name_f, num_f = ri.read_labeled_image_list(dPath_l + dPath_train)
name_test, num_test = ri.read_labeled_image_list(dPath_l + dPath_test)
name_testf, num_testf = ri.read_labeled_image_list(dPath_l + dPath_testf)
name_tests, num_tests, xst, yst = ri.read_labeled_image_list2(dPath_l + dPath_testm)
total_batch = int(num_f / batch_size)

save_path = './validation/v1'
model_path = './model/v1'

restore = False
restore_point = 900000
Checkpoint = model_path + '/cVG iter ' + str(restore_point) + '/'
WeightName = Checkpoint + 'Train_' + str(restore_point) + '.meta'

if restore == False:
    restore_point = 0

saving_iter = 10000
Max_iter = 1000000

# ------- variables

X = tf.placeholder(tf.float32, [batch_size, Height, Width, 3])
Y = tf.placeholder(tf.float32, [batch_size, Height, Width, 3])

MASK = tf.placeholder(tf.float32, [batch_size, Height, Width, 3])
IT = tf.placeholder(tf.float32)

# ------- structure

input = tf.concat([X, MASK], 3)

vec_en = mm.encoder(input, reuse=False, name='G_en')

vec_con = mm.contextual_block(vec_en, vec_en, MASK, 3, 50.0, 'CB1', stride=1)

I_co = mm.decoder(vec_en, Height, reuse=False, name='G_de')
I_ge = mm.decoder(vec_con, Height, reuse=True, name='G_de')

image_result = I_ge * (1-MASK) + Y*MASK

D_real_red = mm.discriminator_red(Y, reuse=False, name='disc_red')
D_fake_red = mm.discriminator_red(image_result, reuse=True, name='disc_red')

# ------- Loss

Loss_D_red = tf.reduce_mean(tf.nn.relu(1+D_fake_red)) + tf.reduce_mean(tf.nn.relu(1-D_real_red))

Loss_D = Loss_D_red

Loss_gan_red = -tf.reduce_mean(D_fake_red)

Loss_gan = Loss_gan_red

Loss_s_re = tf.reduce_mean(tf.abs(I_ge - Y))
Loss_hat = tf.reduce_mean(tf.abs(I_co - Y))

A = tf.image.rgb_to_yuv((image_result+1)/2.0)
A_Y = tf.to_int32(A[:, :, :, 0:1]*255.0)

B = tf.image.rgb_to_yuv((Y+1)/2.0)
B_Y = tf.to_int32(B[:, :, :, 0:1]*255.0)

ssim = tf.reduce_mean(tf.image.ssim(A_Y, B_Y, 255.0))

alpha = IT/Max_iter

Loss_G = 0.1*Loss_gan + 10*Loss_s_re + 5*(1-alpha) * Loss_hat

# --------------------- variable & optimizer

var_D = [v for v in tf.global_variables() if v.name.startswith('disc_red')]
var_G = [v for v in tf.global_variables() if v.name.startswith('G_en') or v.name.startswith('G_de') or v.name.startswith('CB1')]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    optimize_D = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.5, beta2=0.9).minimize(Loss_D, var_list=var_D)
    optimize_G = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(Loss_G, var_list=var_G)

# --------- Run

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = False

sess = tf.Session(config=config)

init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()

if restore == True:
    print('Weight Restoring.....')
    Restore = tf.train.import_meta_graph(WeightName)
    Restore.restore(sess, tf.train.latest_checkpoint(Checkpoint))
    print('Weight Restoring Finish!')

start_time = time.time()
for iter_count in range(restore_point, Max_iter + 1):

    i = iter_count % total_batch
    e = iter_count // total_batch

    if i == 0:
        np.random.shuffle(name_f)

    data_g = ri.MakeImageBlock(name_f, Height, Width, i, batch_size)

    data_temp = 255.0 * ((data_g + 1) / 2.0)

    mask = op.ff_mask_batch(Height, batch_size, 50, 30, 3.14, 5, 15)

    data_m = data_temp * mask

    data_m = (data_m / 255.0) * 2.0 - 1

    _, Loss1 = sess.run([optimize_D, Loss_D], feed_dict={X: data_m, Y: data_g, MASK: mask})
    _, Loss2, Loss3 = sess.run([optimize_G, Loss_G, Loss_s_re], feed_dict={X: data_m, Y: data_g, MASK: mask, IT:iter_count})

    if iter_count % 100 == 0:
        consume_time = time.time() - start_time
        print('%d     Epoch : %d       D Loss = %.5f    G Loss = %.5f    Recon Loss = %.5f     time = %.4f' % (iter_count, e, Loss1, Loss2, Loss3, consume_time))
        start_time = time.time()

    if iter_count % saving_iter == 0:

        print('SAVING MODEL')
        Temp = model_path + '/cVG iter %s/' % iter_count

        if not os.path.exists(Temp):
            os.makedirs(Temp)

        SaveName = (Temp + 'Train_%s' % (iter_count))
        saver.save(sess, SaveName)
        print('SAVING MODEL Finish')

        psnr_l = 0
        psnr_g = 0
        psnr_f = 0
        ssim_m = 0
        num_s = random.sample(range(num_test - batch_size), 10)
        for isave in range(5):
            mask_sizet = random.randint(64, 128)

            data_test = ri.MakeImageBlock(name_test, Height, Width, num_s[isave]//batch_size, batch_size)
            data_tempt = 255.0 * ((data_test + 1) / 2.0)
            mask_ts, xs, ys = op.make_sq_mask(Height, mask_sizet, batch_size)
            mask_tf = op.ff_mask(Height, batch_size, 50, 20, 3.14, 6, 10)

            data_tempts = data_tempt * mask_ts
            data_mts = (data_tempts / 255.0) * 2.0 - 1

            data_temptf = data_tempt * mask_tf
            data_mtf = (data_temptf / 255.0) * 2.0 - 1

            img_sample = sess.run(image_result, feed_dict={X: data_mts, Y: data_test, MASK: mask_ts})
            img_sample2 = sess.run(image_result, feed_dict={X: data_mtf, Y: data_test, MASK: mask_tf})

            for kk in range(batch_size):
                temp_img1 = img_sample[kk,:,:,:]
                temp_img2 = data_test[kk,:,:,:]
                temp_img3 = img_sample2[kk, :, :, :]

                img_gt = 255.0 * ((temp_img2 + 1) / 2.0)
                img_ge = 255.0 * ((temp_img1 + 1) / 2.0)
                img_ge2 = 255.0 * ((temp_img3 + 1) / 2.0)

                Bigpaper1 = np.zeros((Height, 3 * Width + 60, 3))
                Bigpaper1[0:Height, 0:Width, :] = img_gt
                Bigpaper1[0:Height, Width + 30: 2 * Width + 30, :] = data_tempts[kk,:,:,:]
                Bigpaper1[0: Height, 2 * Width + 60: 3 * Width + 60, :] = img_ge

                Bigpaper2 = np.zeros((Height, 3 * Width + 60, 3))
                Bigpaper2[0:Height, 0:Width, :] = img_gt
                Bigpaper2[0:Height, Width + 30: 2 * Width + 30, :] = data_temptf[kk, :, :, :]
                Bigpaper2[0: Height, 2 * Width + 60: 3 * Width + 60, :] = img_ge2

                save_name = save_path + '/%04d' % iter_count
                name = save_name + '/img_%02d_s.png' % (isave * batch_size + kk)
                name2 = save_name + '/img_%02d_f.png' % (isave * batch_size + kk)
                if not os.path.exists(save_name):
                    os.makedirs(save_name)
                sci.imsave(name, Bigpaper1)
                sci.imsave(name2, Bigpaper2)

        for ipsnr in range(100):
            mask_sizep = 128

            data_test = ri.MakeImageBlock(name_test, Height, Width, ipsnr, batch_size)
            data_tempt = 255.0 * ((data_test + 1) / 2.0)
            mask_t = ri.MakeImageBlock(name_tests, Height, Width, ipsnr, batch_size)
            mask_t = (mask_t + 1) / 2

            data_tempt = data_tempt * mask_t
            data_mt = (data_tempt / 255.0) * 2.0 - 1

            img_sample1, ssim_temp = sess.run([image_result, ssim], feed_dict={X: data_mt, Y: data_test, MASK: mask_t})

            for kk in range(batch_size):
                xx = int(xst[ipsnr * batch_size + kk])
                yy = int(yst[ipsnr * batch_size + kk])
                img_sample2 = img_sample1[:, xx:xx + mask_sizep, yy:yy + mask_sizep, :]
                img_sample3 = data_test[:, xx:xx + mask_sizep, yy:yy + mask_sizep, :]

                temp_img1 = img_sample1[kk,:,:,:]
                temp_img2 = img_sample2[kk,:,:,:]
                temp_img3 = data_test[kk,:,:,:]
                temp_img4 = img_sample3[kk,:,:,:]

                img_re = 255.0 * ((temp_img1 + 1) / 2.0)
                img_rem = 255.0 * ((temp_img2 + 1) / 2.0)
                img_gt = 255.0 * ((temp_img3 + 1) / 2.0)
                img_gtm = 255.0 * ((temp_img4 + 1) / 2.0)

                mse_l = np.mean(np.square(img_gtm - img_rem))
                mse_g = np.mean(np.square(img_gt - img_re))
                psnr_l += 10 * np.log10(255.0 * 255.0 / mse_l)
                psnr_g += 10 * np.log10(255.0 * 255.0 / mse_g)
            ssim_m += ssim_temp

        print('\nLocal = ', '%.4f' % (psnr_l/800),'\nGlobal = ', '%.4f\n' % (psnr_g/800), 'ssim = %.4f\n' % (ssim_m/100.0))

        pp = open(save_path + '/PSNR_log.txt', 'a+')
        data = '--------------------' + '\n%d' % iter_count + '\nLocal = ' + '%.4f' % (psnr_l / 800) + '\nGlobal = ' + '%.4f\n' % (psnr_g / 800) + 'ssim = %.4f\n' % (ssim_m/100.0)

        pp.write(data)
        pp.close()

        for ipsnr in range(100):

            data_test = ri.MakeImageBlock(name_test, Height, Width, ipsnr, batch_size)
            data_tempt = 255.0 * ((data_test + 1) / 2.0)
            mask_t = ri.MakeImageBlock(name_testf, Height, Width, ipsnr, batch_size)
            mask_t = (mask_t + 1) / 2

            data_tempt = data_tempt * mask_t
            data_mt = (data_tempt / 255.0) * 2.0 - 1

            img_sample1, ssim_temp = sess.run([image_result, ssim], feed_dict={X: data_mt, Y: data_test, MASK: mask_t})

            for kk in range(batch_size):

                temp_img1 = img_sample1[kk,:,:,:]
                temp_img3 = data_test[kk,:,:,:]

                img_re = 255.0 * ((temp_img1 + 1) / 2.0)
                img_gt = 255.0 * ((temp_img3 + 1) / 2.0)

                mse = np.mean(np.square(img_gt - img_re))
                psnr_f += 10 * np.log10(255.0 * 255.0 / mse)


        print('\nPSNR_f = ', '%.4f\n' % (psnr_f/800))

        pp = open(save_path + '/PSNR_log.txt', 'a+')
        data = '\nPSNR_f = ' + '%.4f\n' % (psnr_f / 800)

        pp.write(data)
        pp.close()