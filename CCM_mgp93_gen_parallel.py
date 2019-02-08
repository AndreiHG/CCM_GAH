import multiprocessing as mp
import data_handle_lib as dh
import pandas as pd
import numpy as np
import time as tm
import CCM_GAH_lib as ccm

if __name__ == '__main__':
    print("Number of CPUs: %s"%(mp.cpu_count()))

    # Read mgp93 data
    data_path = 'data/mgp93_data_genus.tsv'
    metadata_path = 'data/mgp93_metadata.csv'

    raw_data, raw_metadata, metadata_descr = dh.read_raw_data(data_path, metadata_path)
    df_mgp93 = dh.merge_data(raw_data, raw_metadata)

    # Select data
    subject = "F4"  # M3 or F4
    sample_site = "feces"  # feces, L_palm, R_palm, Tongue

    # Select data for a certain subject and sample location
    df_data = df_mgp93[((df_mgp93.host_individual == subject) | (df_mgp93['host_individual'].isnull()))
                       & ((df_mgp93.common_sample_site == sample_site) | (df_mgp93['common_sample_site'].isnull()))]

    how_long_metadata = np.count_nonzero(np.isnan(df_data.index.values))
    first_day = int(df_data.index.values[how_long_metadata])
    first_species = 0  # column index of first species (usually 0)

    data_range = df_data.columns.values[df_data.columns.get_loc(first_species):].astype(int)
    how_many_species = 5
    only_most_abundant = True

    # Select only the most abundant
    if (only_most_abundant):
        total_population = df_data.loc[first_day:, data_range].sum(axis=0)
        data_range = (total_population.sort_values(ascending=False).index.values).astype(int)[0:how_many_species]

    df_data = pd.concat([df_data.iloc[:, 0:df_data.columns.get_loc(first_species) - 1], df_data.loc[:, data_range]],
                        axis=1)

    df_data_norm = df_data.copy()
    df_data_norm.loc[first_day:, data_range] = dh.df_normalize(df_data_norm.loc[first_day:, data_range], along_row=True)

    time = df_data_norm.loc[first_day:, data_range].index.values

    bacteria_IDs = df_data.columns.values[4:]
    timestr = tm.strftime("%Y%m%d_%H%M%S")

    # Create the empty file and the header. Next we just append to it
    df_result = pd.DataFrame(
        {"x_ID": [], "y_ID": [], "x_name": [], "y_name": [], "spearman_coeff": [], "spearman_coeff_p": [],
         "pearson_coeff": [], "pearson_coeff_last": [], "L": [], "L_final": [], "L_step": [],
         "subject": [], "sample_loc": [], "E": []})
    df_result.to_csv("mgp93_" + subject + "_" + sample_site + "_CCMed_parallel_" + timestr + ".csv")

    # Do CCM on all possible combinations of the selected species
    # Append to the previously created csv file live, but in chunks

    ########
    # Series
    ########

    # total_start_time = tm.time()
    # for m in range(len(bacteria_IDs)):
    #     start_time = tm.time()
    #     for n in np.arange(m + 1, len(bacteria_IDs)):
    #         df_result = df_result.append(ccm.single_CCM(df_data_norm, bacteria_IDs[m], bacteria_IDs[n], \
    #                                                     L_step=20, print_timeit=False, E=7), sort=False)
    #     #end_time = tm.time()
    #     #print("Appended! Finished lot %s in %.0f s." % (m + 1, end_time - start_time))
    #     #df_result.to_csv("mgp93_" + subject + "_" + sample_site + "_CCMed_series_" + timestr +".csv", header=None, mode="a")
    #     #df_result.drop(df_result.index, inplace=True)  # we already output it
    #
    # print("Total time was %.2f seconds." %(tm.time()-total_start_time))

    ##########
    # Parallel
    ##########
    total_start_time = tm.time()
    # After each CCM, append to output file (only happens if output file name is provided in the single_CCM function)
    output_file = "mgp93_" + subject + "_" + sample_site + "_CCMed_parallel_" + timestr + ".csv"
    df_result = pd.DataFrame({"x_ID": [], "y_ID": [], "x_name": [], "y_name": [],
                              "spearman_coeff": [], "spearman_coeff_p": [],
                              "pearson_coeff": [], "pearson_coeff_last": [],
                              "L": [], "L_final": [], "L_step": [],
                              "subject": [], "sample_loc": [], "E": []})
    df_result.to_csv(output_file)
    # First build the argument list for the parallel computation
    args_parallel = []
    for m in range(len(bacteria_IDs)):
        for n in np.arange(m+1, len(bacteria_IDs)):
            # arguments for single_CCM
            # df, x_ID, y_ID, L_step, E, taxonomy,
            # print_timeit, print_results, plot_result):
            args_parallel.append([df_data_norm, bacteria_IDs[m], bacteria_IDs[n], 1, 10, "genus",
                                  False, False, False, output_file])

    with mp.Pool(processes=6) as pool:
        pool.starmap(ccm.single_CCM, args_parallel)

    print("Total time for parallel with %.0f processes was %.2f seconds." % (6, tm.time() - total_start_time))

    #df_result.to_csv("mgp93_" + subject + "_" + sample_site + "_CCMed_parallel_" + timestr + ".csv",
    #                 header=None, mode="a")