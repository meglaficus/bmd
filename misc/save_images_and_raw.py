from scipy.ndimage.interpolation import zoom
import numpy as np
import cv2
import os

from misc.utils import read_dicom, grayscale_to_rgb


def save_images(args, pt_name, a, label_info_array):

    results_all_dir = args.results_dir + "All/" + pt_name + "/"
    results_hit_dir = args.results_dir + "Hit/" + pt_name + "/"
    results_center_dir = args.results_dir + "Center/" + pt_name + "/"
    os.makedirs(results_all_dir, exist_ok=True)
    os.makedirs(results_hit_dir, exist_ok=True)
    os.makedirs(results_center_dir, exist_ok=True)

    print(pt_name, "saving images...")

    L = a.width

    a.bs_rgb = grayscale_to_rgb(a.bs*255)

    a.ct_with_bbs_96 = np.copy(a.ct_rgb)
    a.ct_with_bbs_96[:, :, :, 0:2] += a.bbs_96[:, :, :, :] * 25

    a.ct_with_bbs = np.copy(a.ct_rgb)
    a.ct_with_bbs[:, :, :, 0:2] += a.bbs[:, :, :, :] * 25

    a.seg_0or1_rgb = np.concatenate([np.zeros_like(a.seg_0or1), a.seg_0or1, a.seg_0or1], axis=3) * 255

    a.seg_jet = np.zeros_like(a.ct_rgb)
    a.seg = (a.seg * 255).astype("uint8")
    for z in range(a.seg.shape[0]):
        jet = cv2.applyColorMap(a.seg[z], cv2.COLORMAP_JET)
        a.seg_jet[z] = jet
    a.seg_jet *= np.where(a.seg > 5, 1, 0)

    a.ct_rgb_with_pred += a.seg_0or1_rgb // 2
    a.ct_rgb_with_pred -= a.ct_rgb * a.seg_0or1 * 0.5
    a.ct_rgb_with_pred = np.where(a.ct_rgb_with_pred > 255, 255, a.ct_rgb_with_pred)

    a.ct_rgb_with_pred_and_label = np.copy(a.ct_rgb_with_pred)
    a.ct_rgb_with_pred_and_label[:, :, :, 0] += a.lb_0or1[:, :, :, 0] * 255
    a.ct_rgb_with_pred_and_label[:, :, :, 2] += a.lb_0or1[:, :, :, 0] * 255
    a.ct_rgb_with_pred_and_label[:, :, :, 1] -= a.lb_0or1[:, :, :, 0] * 1000
    a.ct_rgb_with_pred_and_label = np.where(a.ct_rgb_with_pred_and_label > 255, 255, a.ct_rgb_with_pred_and_label)
    a.ct_rgb_with_pred_and_label = np.where(a.ct_rgb_with_pred_and_label < 0, 64, a.ct_rgb_with_pred_and_label)

    a.dummy = np.zeros_like(a.ct_rgb)
    a.stacked_image_0 = np.zeros((50, a.ct.shape[2]*4, 3))

    ### Save axial images ###
    z_list = np.arange(a.margin_z, a.height - a.margin_z, 4).tolist()

    gt_bar = np.zeros((len(z_list), 1, 3))
    bb_bar = np.zeros((len(z_list), 1, 3))
    seg_bar = np.zeros((len(z_list), 1, 3))

    for i, z in enumerate(z_list):
        if a.seg_0or1[z].sum() > 0:
            seg_bar[i, :, 1:3] = 255
        if a.seg[z].sum() > 0:
            bb_bar[i, :, 0:2] = 255
        if np.unique(a.lb[z]).sum() > 0:
            gt_bar[i, :, 0] = 255
            gt_bar[i, :, 2] = 255

    for i, z in enumerate(z_list):

        a.stacked_image_1 = np.hstack([a.ct_rgb[z], a.bs_rgb[z], a.ct_with_bbs_96[z], a.seg_jet[z]])
        a.stacked_image_2 = np.hstack([a.ct_with_bbs[z], a.seg_0or1_rgb[z], a.ct_rgb_with_pred[z], a.ct_rgb_with_pred_and_label[z]])

        a.stacked_image = np.vstack([a.stacked_image_0,
                                    a.stacked_image_1[int(L * 0.15):int(L * 0.90), :],
                                     a.stacked_image_2[int(L * 0.15):int(L * 0.90), :]]).astype("int16")

        cp_bar = np.zeros((len(z_list), 1, 3))
        cp_bar[i, :, :] = 255

        seg_bar_resized = cv2.resize(seg_bar, (10, a.stacked_image.shape[0]))
        bb_bar_resized = cv2.resize(bb_bar, (10, a.stacked_image.shape[0]))
        gt_bar_resized = cv2.resize(gt_bar, (10, a.stacked_image.shape[0]))
        cp_bar_resized = cv2.resize(cp_bar, (10, a.stacked_image.shape[0]))

        a.stacked_image = np.hstack(
            [a.stacked_image, cp_bar_resized, seg_bar_resized, bb_bar_resized, gt_bar_resized])

        if a.seg_labeled[z].sum() > 0:
            lb_ids = np.unique(a.seg_labeled[z]).tolist()
            del lb_ids[0]
            label_info = ""
            for lb_id in lb_ids:
                volume = int(label_info_array[lb_id-1][1])
                conf_level = str(round(label_info_array[lb_id-1][2], 3)) + "-" + str(round(label_info_array[lb_id-1][3], 3)) + "-" + str(round(label_info_array[lb_id-1][4], 3))
                label_info += str(lb_id) + " " + str(volume) + " " + conf_level + "   "
            cv2.putText(a.stacked_image, label_info, (15, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), lineType=cv2.LINE_AA)

        image_path = results_all_dir + pt_name + "_axi" + str(z).zfill(4) + ".jpg"
        cv2.imwrite(image_path, a.stacked_image)

        if a.seg_0or1[z].sum() > 0 or a.lb_0or1[z].sum() > 0:
            image_path = results_hit_dir + pt_name + "_axi" + str(z).zfill(4) + ".jpg"
            cv2.imwrite(image_path, a.stacked_image)

    ### Save center of lesions ###
    id_list = np.unique(a.lb).tolist()
    del id_list[0]
    for i in id_list:
        r_array = np.where(a.lb == i, 1, 0)
        arr = np.nonzero(r_array)
        z = int(arr[0].mean())

        a.stacked_image_1 = np.hstack([a.ct_rgb[z], a.ct_with_bbs_96[z], a.seg_jet[z], a.ct_with_bbs[z]])
        a.stacked_image_2 = np.hstack([a.dummy[z], a.seg_0or1_rgb[z], a.ct_rgb_with_pred[z], a.ct_rgb_with_pred_and_label[z]])
        a.stacked_image = np.vstack([a.stacked_image_1[int(L * 0.15):int(L * 0.90), :],
                                     a.stacked_image_2[int(L * 0.15):int(L * 0.90), :]]).astype("int16")

        if a.seg_labeled[z].sum() > 0:
            lb_ids = np.unique(a.seg_labeled[z]).tolist()
            del lb_ids[0]
            label_info = ""
            for lb_id in lb_ids:
                volume = int(label_info_array[lb_id-1][1])
                conf_level = str(round(label_info_array[lb_id-1][2], 3)) + "-" + str(round(label_info_array[lb_id-1][3], 3)) + "-" + str(round(label_info_array[lb_id-1][4], 3))
                label_info += str(lb_id) + " " + str(volume) + " " + conf_level + "   "
            cv2.putText(a.stacked_image, label_info, (15, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), lineType=cv2.LINE_AA)

        image_path = results_center_dir + pt_name + "_Lesion" + str(i).zfill(3) + ".jpg"
        cv2.imwrite(image_path, a.stacked_image)

    ### Save coronal images ###
    y_list = np.arange(a.margin_xy, a.width - a.margin_xy, 4).tolist()

    gt_bar = np.zeros((len(y_list), 1, 3))
    bb_bar = np.zeros((len(y_list), 1, 3))

    for i, y in enumerate(y_list):
        if a.seg_0or1[:, y, :].sum() > 0:
            bb_bar[i, :, 1:3] = 255
        if np.unique(a.lb[:, y, :]).sum() > 0:
            gt_bar[i, :, 0] = 255
            gt_bar[i, :, 2] = 255


    for i, y in enumerate(y_list):
        a.stacked_image = np.hstack([a.ct_rgb[:, y, :, :], a.ct_rgb_with_pred[:, y, :, :],
                                     a.ct_rgb_with_pred_and_label[:, y, :, :]]).astype("int16")

        cp_bar = np.zeros((len(y_list), 1, 3))
        cp_bar[i, :, :] = 255

        bb_bar_resized = cv2.resize(bb_bar, (10, a.stacked_image.shape[0]))
        gt_bar_resized = cv2.resize(gt_bar, (10, a.stacked_image.shape[0]))
        cp_bar_resized = cv2.resize(cp_bar, (10, a.stacked_image.shape[0]))

        a.stacked_image = np.hstack([a.stacked_image, cp_bar_resized, bb_bar_resized, gt_bar_resized])

        image_path = results_all_dir + pt_name + "_cor" + str(y).zfill(4) + ".jpg"
        cv2.imwrite(image_path, a.stacked_image)

    return


def save_raw(args, pt_name, a):

    print("### Save raw file ###")
    l_array = a.seg_0or1[:,:,:,0]

    dcm_dir = args.dicom_dir + pt_name + "/"
    results_raws_dir = args.results_dir + "PRED/"
    os.makedirs(results_raws_dir, exist_ok=True)
    raw_path = results_raws_dir + pt_name + "_pred.raw"
    text_path = results_raws_dir + pt_name + "_pred.txt"

    d_array = read_dicom(dcm_dir)

    xxx = d_array.shape[2] / l_array.shape[2]
    yyy = d_array.shape[1] / l_array.shape[1]
    zzz = d_array.shape[0] / l_array.shape[0]

    print(l_array.shape, " > ", d_array.shape)

    if zzz == 1:
        print("2D zooming") # much faster than 3D zooming
        for z in range(d_array.shape[0]):
            if z == 0:
                l_array_post = np.zeros_like(d_array)
            l_array_tmp = zoom(l_array[z], zoom=[yyy, xxx])
            l_array_post[z] = l_array_tmp
        l_array = l_array_post
    else:
        print("3D zooming")
        l_array = zoom(l_array, zoom=[zzz, yyy, xxx])

    (l_array * 255).astype("uint8").tofile(raw_path)

    l_shape_str = str(int(l_array.shape[0])) + "," + str(int(l_array.shape[1])) + "," + str(int(l_array.shape[2]))
    print(l_shape_str)
    with open(text_path, mode='w') as f:
        f.write(l_shape_str)

    return