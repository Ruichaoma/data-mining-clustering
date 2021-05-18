import math
import sys
from sklearn.cluster import KMeans
import numpy as np
import time
import random
import json
import pandas

def ds_creation(ii, second, third):
    ds_group_set[ii] = {}
    ds_group_set[ii][0] = []
    for point in second:
        ds_group_set[ii][0].append(index_dict[point])
    lennn = len(ds_group_set[ii][0])
    ds_group_set[ii][1] = lennn
    ds_group_set[ii][2] = np.sum(third[second, :].astype(np.float), axis=0)
    ds_group_set[ii][3] = np.sum((third[second, :].astype(np.float)) * (third[second, :].astype(np.float)), axis=0)
    ds_group_set[ii][4] = np.sqrt((ds_group_set[ii][3][:] / ds_group_set[ii][1]) - (
            ds_group_set[ii][2][:] * ds_group_set[ii][2][:] / (ds_group_set[ii][1] * ds_group_set[ii][1])))
    ds_group_set[ii][5] = ds_group_set[ii][2] / ds_group_set[ii][1]


def cs_creation(ii, second, third):
    cs_cluster_set[ii] = {}
    cs_cluster_set[ii][0] = []
    for point in second:
        pointid = list(rs_dict.keys())[list(rs_dict.values()).index(rs_point_lst[point])]
        cs_cluster_set[ii][0].append(pointid)
    lennnn = len(cs_cluster_set[ii][0])
    cs_cluster_set[ii][1] = lennnn
    cs_cluster_set[ii][2] = np.sum(third[second, :].astype(np.float), axis=0)
    cs_cluster_set[ii][3] = np.sum((third[second, :].astype(np.float)) * (third[second, :].astype(np.float)), axis=0)
    cs_cluster_set[ii][4] = np.sqrt((cs_cluster_set[ii][3][:] / cs_cluster_set[ii][1]) - (
            cs_cluster_set[ii][2][:] * cs_cluster_set[ii][2][:] / (cs_cluster_set[ii][1] * cs_cluster_set[ii][1])))
    cs_cluster_set[ii][5] = cs_cluster_set[ii][2] / cs_cluster_set[ii][1]


def ds_increment(new_list, position, updated_point, centroid_used):
    new_list[centroid_used][0].append(position)
    new_list[centroid_used][1] += 1
    for i in range(0, len_point):
        new_list[centroid_used][2][i] += updated_point[i]
        new_list[centroid_used][3][i] += updated_point[i] * updated_point[i]
    new_list[centroid_used][4] = np.sqrt((new_list[centroid_used][3][:] / new_list[centroid_used][1]) - (
            (new_list[centroid_used][2][:]) * (new_list[centroid_used][2][:]) / (
                new_list[centroid_used][1] * new_list[centroid_used][1])))
    new_list[centroid_used][5] = new_list[centroid_used][2] / new_list[centroid_used][1]


def cluster_formation(clusters):
    new_dict = {}
    for i in range(len(clusters)):
        clusterid = clusters[i]
        if clusterid in new_dict:
            new_dict[clusterid].append(i)
        else:
            new_dict[clusterid] = [i]
    return new_dict


def transfer_dict_formation(hash_into):
    cluster_no_rs = new_kmeans.fit_predict(hash_into)
    cluster_formation_no_rs = {}
    len_cluster_no_rs = len(cluster_no_rs)
    for i in range(0, len_cluster_no_rs):
        point_no_rs = hash_into[i]
        cluster_belong_no_rs = cluster_no_rs[i]
        if cluster_belong_no_rs not in cluster_formation_no_rs:
            cluster_formation_no_rs[cluster_belong_no_rs] = [point_no_rs]
        else:
            cluster_formation_no_rs[cluster_belong_no_rs].append(point_no_rs)


def judgepoint_mahalanobis(point, sets):
    initial_nearest_threshold = parameter_2d
    index_near_cluster = -10000
    for indexing in sets.keys():
        sd_float, centroid_float = sets[indexing][4].astype(np.float), sets[indexing][5].astype(np.float)
        mah_dis = 0
        for ss in range(0, len_point):
            mah_dis += ((point[ss] - centroid_float[ss]) / sd_float[ss]) * ((point[ss] - centroid_float[ss]) / sd_float[ss])
        mah_dis = np.sqrt(mah_dis)
        if mah_dis < initial_nearest_threshold:
            initial_nearest_threshold = mah_dis
            index_near_cluster = indexing
    return index_near_cluster


def centroid_calculation(summ, avgg, number_point):
    centroidd = summ / avgg
    wanted_centroid = centroidd ** 2 + np.sqrt(centroidd ** 3)
    final_centroid = np.sqrt(wanted_centroid) * number_point
    return final_centroid


def final_cluster_formation(data_dict, specificc=None, countingg=None, lennn=None, summm=None, sumsqq=None,
                            centroidd=None):
    specific_lst , countingg_lst, lenn_lst, summ_lst, sumsq_lst,centroid_lst = [],[],[],[],[],[]
    num_point = 0
    for i in range(len(data_dict)):
        specific_lst.append(data_dict[1][0])
        countingg_lst.append(data_dict[1][1])
        lenn_lst.append(data_dict[1][2])
        summ_lst.append(data_dict[1][3])
        sumsq_lst.append(data_dict[1][4])
        centroid_lst.append(data_dict[1][3]/data_dict[1][2])
    my_dict = {}
    for i in range(len(data_dict)):
        my_dict[specificc] = specific_lst[i]
        my_dict[countingg] = countingg_lst[i]
        my_dict[lennn] = lenn_lst[i]
        my_dict[summm] = summ_lst[i]
        my_dict[sumsqq] = sumsq_lst[i]
        my_dict[centroidd] = centroid_lst[i]


def randomly_load(dataset_used):
    df_first = dataset_used.sample(frac = 0.2)
    df_rest = dataset_used.loc[~dataset_used.index.isin(df_first.index)]
    return df_rest


def cs_cs_merging(cs_group1, cs_group2):
    nearest_cluster_dict = {}
    group1_key, group2_key = cs_group1.keys(), cs_group2.keys()
    for i in group1_key:
        initial_group_mah_dis = parameter_2d
        clostest_group = i
        for j in group2_key:
            if i != j:
                initial_mah_dis_group1, initial_mah_dis_group2 = 0, 0
                sd_group1 = cs_group1[i][4]
                sd_group2 = cs_group2[j][4]
                centroid_group1 = cs_group1[i][5]
                centroid_group2 = cs_group2[j][5]
                for k in range(0, len_point):
                    if sd_group1[k] > 0 and sd_group2[k] > 0:
                        initial_mah_dis_group1 += (((centroid_group1[k] - centroid_group2[k]) / sd_group2[k]) * (
                                    (centroid_group1[k] - centroid_group2[k]) / sd_group2[k]))
                        initial_mah_dis_group2 += (((centroid_group2[k] - centroid_group1[k]) / sd_group1[k]) * (
                                    (centroid_group2[k] - centroid_group1[k]) / sd_group1[k]))
                sqrt_initial_mah_dis_group1 = np.sqrt(initial_mah_dis_group1)
                sqrt_initial_mah_dis_group2 = np.sqrt(initial_mah_dis_group2)
                min_mah_dis = min(sqrt_initial_mah_dis_group1, sqrt_initial_mah_dis_group2)
                if min_mah_dis < initial_group_mah_dis:
                    initial_group_mah_dis = min_mah_dis
                    clostest_group = j
        nearest_cluster_dict[i] = clostest_group
    return nearest_cluster_dict


def step12_cs_merging(cs_group1, cs_group2):
    cs_cluster_set[cs_group1][0].extend(cs_cluster_set[cs_group2][0])
    cs_cluster_set[cs_group1][1] = cs_cluster_set[cs_group1][1] + cs_cluster_set[cs_group2][1]
    for i in range(0, len_point):
        cs_cluster_set[cs_group1][2][i] += cs_cluster_set[cs_group2][2][i]
        cs_cluster_set[cs_group1][3][i] += cs_cluster_set[cs_group2][3][i]
        sd_merging = (cs_cluster_set[cs_group1][3][:] / cs_cluster_set[cs_group1][1]) - (
                cs_cluster_set[cs_group1][2][:] * cs_cluster_set[cs_group1][2][:] / (
                    cs_cluster_set[cs_group1][1] * cs_cluster_set[cs_group1][1]))
    cs_cluster_set[cs_group1][4] = np.sqrt(sd_merging)
    cs_cluster_set[cs_group1][5] = cs_cluster_set[cs_group1][2] / cs_cluster_set[cs_group1][1]


def step_12_cs_ds_merging(cs_group, ds_group):
    ds_group_set[ds_group][0].extend(cs_cluster_set[cs_group][0])
    ds_group_set[ds_group][1] = ds_group_set[ds_group][1] + cs_cluster_set[cs_group][1]
    for i in range(0, len_point):
        ds_group_set[ds_group][2][i] += cs_cluster_set[cs_group][2][i]
        ds_group_set[ds_group][3][i] += cs_cluster_set[cs_group][3][i]
        sd_merging = (ds_group_set[ds_group][3][:] / ds_group_set[ds_group][1]) - (
                ds_group_set[ds_group][2][:] * ds_group_set[ds_group][2][:] / (
                ds_group_set[ds_group][1] * ds_group_set[ds_group][1]))
    ds_group_set[ds_group][4] = np.sqrt(sd_merging)
    ds_group_set[ds_group][5] = ds_group_set[ds_group][2] / ds_group_set[ds_group][1]


start_time = time.time()


input_file = sys.argv[1]
cluster = int(sys.argv[2])
output_file = sys.argv[3]


open_ = open(input_file, "r")
data = np.array(open_.readlines())
open_.close()
f = open(output_file, "w")

intermediate_output_count = 0
discard_points_count = 0
compression_cluster_count = 0
compression_point_count = 0
retained_point_count = 0



first_partition = int(len(data) * 0.2)

start_index = 0
end_index = first_partition
first_partition_data = data[start_index:end_index]

index_dict = {}
ppp_dict = {}
first_partition_lst = []

count_count = 0
for row in first_partition_data:
    row = row.split(",")
    index = row[0]
    ppp = row[2:]
    first_partition_lst.append(ppp)
    index_dict[count_count] = index
    ppp_dict[str(ppp)] = index
    count_count += 1

len_point = len(first_partition_lst[0])
parameter_2d = 2 * math.sqrt(len_point)
first_partition_np = np.array(first_partition_lst)


n_cluster_step2 = 5 * cluster
kmeans_model = KMeans(n_clusters=n_cluster_step2, random_state=123)
potential_rs_cluster = kmeans_model.fit_predict(first_partition_np)
rs_cluster = {}

for i in range(len(potential_rs_cluster)):
    point = first_partition_lst[i]
    clusterid = potential_rs_cluster[i]
    if clusterid in rs_cluster:
        rs_cluster[clusterid].append(point)
    else:
        rs_cluster[clusterid] = [point]


rs_dict = {}
for iii in rs_cluster.keys():
    if len(rs_cluster[iii]) == 1:
        point_rscluster = rs_cluster[iii][0]
        dimen = first_partition_lst.index(point_rscluster)
        rs_dict[index_dict[dimen]] = point_rscluster
        first_partition_lst.remove(point_rscluster)
        for l in range(dimen, len(index_dict) - 1):
            index_dict[l] = index_dict[l + 1]


nparray_no_rs = np.array(first_partition_lst)
new_kmeans = KMeans(n_clusters=cluster, random_state=123)
step_4_cluster = cluster_formation(new_kmeans.fit_predict(nparray_no_rs))


ds_group_set = {}
for indexxxx in step_4_cluster.keys():
    ds_creation(indexxxx, step_4_cluster[indexxxx], nparray_no_rs)


rs_point_lst = []
for key in rs_dict.keys():
    rs_point_lst.append(rs_dict[key])

rs_points_array = np.array(rs_point_lst)
step_6_cluster_num = int(len(rs_point_lst) * 0.5 + 1)
step_6kmeans = KMeans(n_clusters=step_6_cluster_num, random_state=123)
cs_clusters = cluster_formation(step_6kmeans.fit_predict(rs_points_array))

cs_cluster_set = {}
for i in cs_clusters.keys():
    if len(cs_clusters[i]) > 1:
        cs_creation(i, cs_clusters[i], rs_points_array)

for ii in cs_clusters.keys():
    if len(cs_clusters[ii]) > 1:
        for i in cs_clusters[ii]:
            deleting = list(rs_dict.keys())[
                list(rs_dict.values()).index(rs_point_lst[i])]
            del rs_dict[deleting]

rs_point_lst = []
for key in rs_dict.keys():
    rs_point_lst.append(rs_dict[key])

f.write("The intermediate results:\n")


for i in ds_group_set.keys():
    discard_points_count += ds_group_set[i][1]
for i in cs_cluster_set.keys():
    compression_cluster_count == compression_cluster_count + 1
    compression_point_count += cs_cluster_set[i][1]
retained_point_count = len(rs_point_lst)
f.write("Round " + str(intermediate_output_count + 1) + ": " + str(discard_points_count) + "," + str(
    compression_cluster_count) + "," + str(
    compression_point_count) + "," + str(retained_point_count) + "\n")


last_round = 4
for num_round in range(1, 5):
    start_index = end_index
    another_data = []
    if num_round == last_round:
        end_index = len(data)
        another_data = data[start_index:end_index]
    else:
        end_index = start_index + first_partition
        another_data = data[start_index:end_index]

    points_lst = []
    last_ctr = count_count
    for row in another_data:
        row = row.split(",")
        index = row[0]
        ppp = row[2:]
        points_lst.append(ppp)
        index_dict[count_count] = index
        ppp_dict[str(ppp)] = index
        count_count = count_count + 1

    nparray_another_data = np.array(points_lst)



    for i in range(len(nparray_another_data)):
        specific_data = nparray_another_data[i]
        new_point = specific_data.astype(np.float)
        new_index = index_dict[last_ctr + i]
        nearest_index_cluster = judgepoint_mahalanobis(new_point, ds_group_set)

        if nearest_index_cluster > -(3 * 1 / 3):
            ds_increment(ds_group_set, new_index, new_point, nearest_index_cluster)
        else:
            nearest_index_cluster = judgepoint_mahalanobis(new_point, cs_cluster_set)
            if nearest_index_cluster > -(3 * 1 / 3):
                ds_increment(cs_cluster_set, new_index, new_point, nearest_index_cluster)
            else:
                rs_dict[new_index] = list(specific_data)
                rs_point_lst.append(list(specific_data))



    nparray_another_data = np.array(rs_point_lst)
    step_11_cluster_num = int(len(rs_point_lst) * 0.5 + 1)
    step_11_kmeans = KMeans(n_clusters=step_11_cluster_num, random_state=123)

    cs_clusters = cluster_formation(step_11_kmeans.fit_predict(nparray_another_data))

    for cscs in cs_clusters.keys():
        if len(cs_clusters[cscs]) > 1:
            counting = 0
            if cscs in cs_cluster_set.keys():
                while counting in cs_cluster_set:
                    counting += 1
            else:
                counting = cscs
            cs_creation(counting, cs_clusters[cscs], nparray_another_data)

    for cscscs in cs_clusters.keys():
        if len(cs_clusters[cscscs]) > 1:
            for i in cs_clusters[cscscs]:
                deleting_point = ppp_dict[str(rs_point_lst[i])]
                if deleting_point in rs_dict.keys():
                    del rs_dict[deleting_point]

    rs_point_lst = []
    for key in rs_dict.keys():
        rs_point_lst.append(rs_dict[key])


    nearest_cluster_dict = cs_cs_merging(cs_cluster_set, cs_cluster_set)

    for step12 in nearest_cluster_dict.keys():
        if step12 != nearest_cluster_dict[step12] and nearest_cluster_dict[
            step12] in cs_cluster_set.keys() and step12 in cs_cluster_set.keys():
            step12_cs_merging(step12, nearest_cluster_dict[step12])
            del cs_cluster_set[nearest_cluster_dict[step12]]


    if num_round == last_round:
        nearest_cluster_dict = cs_cs_merging(cs_cluster_set, ds_group_set)
        for last_run in nearest_cluster_dict.keys():
            if nearest_cluster_dict[last_run] in ds_group_set.keys() and last_run in cs_cluster_set.keys():
                step_12_cs_ds_merging(last_run, nearest_cluster_dict[last_run])
                del cs_cluster_set[last_run]

    discard_points_count = 0
    compression_cluster_count = 0
    compression_point_count = 0
    for key in ds_group_set.keys():
        discard_points_count += ds_group_set[key][1]
    for key in cs_cluster_set.keys():
        compression_cluster_count += 1
        compression_point_count += cs_cluster_set[key][1]
    retained_point_count = len(rs_point_lst)
    f.write("Round " + str(num_round + 1) + ": " + str(discard_points_count) + "," + str(
        compression_cluster_count) + "," + str(
        compression_point_count) + "," + str(retained_point_count) + "\n")

final_point_index_cluster_dict = {}
for point_ds in ds_group_set:
    for single in ds_group_set[point_ds][0]:
        final_point_index_cluster_dict[single] = point_ds

for point_cs in cs_cluster_set:
    for single_cs in cs_cluster_set[point_cs][0]:
        final_point_index_cluster_dict[single_cs] = point_cs

for point_rs in rs_dict:
    final_point_index_cluster_dict[point_rs] = -1

f.write("\n")
f.write("The clustering results: ")
for final_point in sorted(final_point_index_cluster_dict.keys(), key=int):
    f.write("\n")
    f.write(str(final_point) + "," + str(final_point_index_cluster_dict[final_point]))

f.close()

end_time = time.time() - start_time

print("Duration : ", end_time)
