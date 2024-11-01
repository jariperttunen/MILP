import argparse
import numpy as np
import pandas as pd
from pulp import *

def create_lp_problem_lower_upper(file_name:str,sep:str,bmass_constraint_lower:float,bmass_constraint_upper:float):
    """
    LP problem with lower and upper bounds
    """
    # Read the data file
    df = pd.read_csv("Input3.csv", sep=sep)
    #Create column for unique names for each OHJE based on KUVIO and OHJE
    df.insert(1,"UNIQUE_OHJE",df["KUVIO"].apply(int))
    df["UNIQUE_OHJE"]= df["UNIQUE_OHJE"].apply(str)+df["OHJE"].apply(lambda x: "_"+str(x))
    #1.Create the LP problem
    #1.1 All variables are binary, note PuLP will prepend "UNIQUE_OHJE_" prefix in the variable name
    OHJE_var_dict = LpVariable.dicts("UNIQUE_OHJE",df["UNIQUE_OHJE"],cat=LpBinary)
    #1.2 The LP problem: Maximize TOTNPV3  over all variables
    prob = LpProblem(f"Maximize_TOTNPV3_over_all_variables_(UNIQUE_OHJE)", LpMaximize)
    #1.3 The objective function, the first in the LP problem
    prob += lpSum([df["TOTNPV3"][i]*OHJE_var_dict[df["UNIQUE_OHJE"][i]] for i in range(0,len(df["UNIQUE_OHJE"]))])
    #2. Constraints lower and upper
    #2.1 Total TOTbmass20 sum over all variables >= bmass_constraint (1500 for example)
    prob += (lpSum([df["TOTbmass20"][i]*OHJE_var_dict[df["UNIQUE_OHJE"][i]] for i in range(0,len(df["UNIQUE_OHJE"]))]) >= bmass_constraint_lower,
                   "TOTbmass >= " + str(bmass_constraint_lower))
    #2.2 Total TOTbmass20 sum over all variables <= bmass_constraint (2300 for example)
    prob += (lpSum([df["TOTbmass20"][i]*OHJE_var_dict[df["UNIQUE_OHJE"][i]] for i in range(0,len(df["UNIQUE_OHJE"]))]) <= bmass_constraint_upper,
                   "TOTbmass <= " + str(bmass_constraint_upper))
    #2.3 From each group (i.e. stand) select exactly 1 OHJE
    # Group the data by KUVIO. Note: DataFrame indexing remains and the right variable is found via df_group.index[i], not df_group[i]
    df_grouped = df.groupby('KUVIO')
    for (group_name, df_group) in df_grouped:
        prob += (lpSum([OHJE_var_dict[df_group["UNIQUE_OHJE"][df_group.index[i]]] for i in range(0,len(df_group["UNIQUE_OHJE"]))]) == 1, str(group_name)+"_Exactly_one_UNIQUE_OHJE_chosen")
        
    return (prob,df)


def create_lp_problem(file_name:str,sep:str,bmass_constraint:float):
    """
    LP problem with lower bound
    """
    # Read the data file
    df = pd.read_csv("Input3.csv", sep=sep)
    #Create column for unique names for each OHJE based on KUVIO and OHJE
    df.insert(1,"UNIQUE_OHJE",df["KUVIO"].apply(int))
    df["UNIQUE_OHJE"]= df["UNIQUE_OHJE"].apply(str)+df["OHJE"].apply(lambda x: "_"+str(x))
    #1.Create the LP problem
    #1.1 All variables are binary, note PuLP will prepend "UNIQUE_OHJE_" prefix in the variable name
    OHJE_var_dict = LpVariable.dicts("UNIQUE_OHJE",df["UNIQUE_OHJE"],cat=LpBinary)
    #1.2 The LP problem: Maximize TOTNPV3  over all variables
    prob = LpProblem(f"Maximize_TOTNPV3_over_all_variables_(UNIQUE_OHJE)", LpMaximize)
    #1.3 The objective function, the first in the LP problem
    prob += lpSum([df["TOTNPV3"][i]*OHJE_var_dict[df["UNIQUE_OHJE"][i]] for i in range(0,len(df["UNIQUE_OHJE"]))])
    #2. Constraints
    #2.1 Total TOTbmass20 sum over all variables >= bmass_constraint (330 for example)
    prob += (lpSum([df["TOTbmass20"][i]*OHJE_var_dict[df["UNIQUE_OHJE"][i]] for i in range(0,len(df["UNIQUE_OHJE"]))]) >= bmass_constraint,
                   "TOTbmass >= " + str(bmass_constraint))
    #2.2 From each group (i.e. stand) select exactly 1 OHJE
    # Group the data by KUVIO. Note: DataFrame indexing remains and the right variable is found via df_group.index[i], not df_group[i]
    df_grouped = df.groupby('KUVIO')
    for (group_name, df_group) in df_grouped:
        prob += (lpSum([OHJE_var_dict[df_group["UNIQUE_OHJE"][df_group.index[i]]] for i in range(0,len(df_group["UNIQUE_OHJE"]))]) == 1, str(group_name)+"_Exactly_one_UNIQUE_OHJE_chosen")
        
    return (prob,df)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file",dest="i",type=str,required=True,help="Input csv file")
    parser.add_argument("-s","--separator",dest="s",type=str,default=';',help="Separator in csv input file (default ';')")
    parser.add_argument("-r","--excel_file",dest="r",type=str,required=True,help="Write the LP solution and selected variables to Excel file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-c","--constraint",dest="c",type=float,help="TOTbmass20 constraint (see the input csv file)")
    group.add_argument("--clower",dest="clower",type=float,help="TOTbmass20 lower bound constraint  (see the input csv file)")
    parser.add_argument("--cupper",dest="cupper",type=float,help="TOTbmass20 lower bound constraint  (see the input csv file)")
    parser.add_argument("-m","--lpm",dest="m",help="Write LP model to file")
    parser.add_argument("-l","--log",dest="l",help="Write Log file")
    args = parser.parse_args()

    print("Creating LP problem from:",args.i)
    prob=None
    df=None
    if args.c:
        (prob,df) = create_lp_problem(args.i,args.s,args.c)
    else:
        (prob,df) = create_lp_problem_lower_upper(args.i,args.s,args.clower,args.cupper)
    if args.m:
        print("Writing LP problem to:", args.m)
        prob.writeLP(args.m)
    print("Solving LP problem")
    if args.l:
        print("Log file:",args.l)
        prob.solve(pulp.PULP_CBC_CMD(logPath=args.l))
    else:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    print("Done")
    solution = value(prob.objective)
    status = LpStatus[prob.status]
    #Collect results to Excel
    #Problem as dictionary has a lot of information
    print("Preparing data for the result Excel file")
    d = prob.toDict()
    #Select variables
    var_ls=d['variables']
    #Select variables part of solution
    var_solution_ls = [v for v in var_ls if v['varValue']==1.0]
    var_name_ls = [v['name'] for v in var_solution_ls]
    #Remove PuLP generated prefix from variable name
    var_unique_ls = [s.removeprefix('UNIQUE_OHJE_') for s in var_name_ls]
    #Select solution from the data frame (inout data)
    df_result_rows = df.loc[df["UNIQUE_OHJE"].isin(var_unique_ls)]
    df_result_rows = df_result_rows.reset_index(drop=True)
    #Calculate totbmass20 in solution (should be:  totbmass20 >= bmass_constraint)
    totbmass20 = df_result_rows['TOTbmass20'].sum()
    summary_ls=None
    df_summary=None
    if args.c:
        #Seems default is 4 by 1 dataframe, transpose to 1 by 4.
        summary_ls = [solution,status,totbmass20,args.c]
        df_summary = pd.DataFrame(summary_ls).T
        df_summary.columns=["SOLUTION_TOTNPV3","STATUS","TOTbmass20_SUM","TOTbmass20_CONSTRAINT"]
    else:
        summary_ls = [solution,status,totbmass20,args.clower,args.cupper]
        df_summary = pd.DataFrame(summary_ls).T
        df_summary.columns=["SOLUTION_TOTNPV3","STATUS","TOTbmass20_SUM","TOTbmass20_LOWER_CONSTRAINT","TOTbmass20_UPPER_CONSTRAINT"]
    #Concatinate dataframes side by side
    df_result = pd.concat([df_result_rows,df_summary],axis=1,join='outer')
    print("Done")
    print("Writing results to:",args.r)
    df_result.to_excel(args.r)
    print("Done")
    
