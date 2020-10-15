import numpy as np
import argparse

# Initialize
def Initialize(population_size, number_of_parameters):
    population = np.random.randint(2, size=(population_size, number_of_parameters))

    return population
    
# Evaluation
def Evaluate(chromosome, type_function, k = -9999):
    global number_of_evaluations
    number_of_evaluations += 1

    if type_function == 'oneMax':
        return oneMaxEvaluation(chromosome)

    elif type_function == 'trap5':
        return trapEvaluation(chromosome, k)
    else:
        print('Error type of function')
        return

def oneMaxEvaluation(chromosome):
    return np.sum(chromosome)

def trapEvaluation(chromosome, k):
    if len(chromosome) % k != 0:
        print("Error in evaluating deceptive trap k: Number of parameters is not a multiple of k.")
        return

    m = int(len(chromosome) / k)
    score = 0
    for i in range(m):
        u = 0
        for j in range(k):
            u += chromosome[i*k+j]
        if u == k:
            score += u
        else:
            score += k - u - 1

    return score

# Crossover
def Crossover(population, number_of_parameters, type_cross):
    offspring = []
    tmp_population = population.copy()
    np.random.shuffle(tmp_population)

    # print(tmp_population)
    if type_cross == '1X':
        for i in range(0, len(population), 2):
            cross_point = np.random.randint(0, number_of_parameters + 1)
            parent_1 = tmp_population[i]
            parent_2 = tmp_population[i + 1]
            offspring_1 = np.concatenate([parent_1[:cross_point], parent_2[cross_point:]])
            offspring_2 = np.concatenate([parent_2[:cross_point], parent_1[cross_point:]])
            offspring.append(offspring_1)
            offspring.append(offspring_2)
    
    elif type_cross == 'UX':
        for i in range(0, len(population), 2):
            parent_1 = tmp_population[i]
            parent_2 = tmp_population[i + 1]
            offspring_1 = []
            offspring_2 = []

            probability = np.random.uniform(0, 1, size=number_of_parameters)
            for prob, p1, p2 in zip(probability, parent_1, parent_2):
                # prob <= 0.5 -> offspring_1 = parent_1, offspring_2 = parent_2
                # prob <  0.5 -> offspring_1 = parent_2, offspring_2 = parent_1
                if prob <= 0.5:
                    offspring_1.append(p1)
                    offspring_2.append(p2)
                else:
                    offspring_1.append(p2)
                    offspring_2.append(p1)
            
            offspring.append(offspring_1)
            offspring.append(offspring_2)

    else:
        print('Error type of crossover')

    return np.array(offspring)


# P + O Pool
def Popop(population, offspring):
    return np.concatenate([population, offspring])

# Tournament selection
def TournamentSelection(population, tournament_size, type_function, k=-9999):
    if len(population) % tournament_size != 0:
        print('Error: Size of population is not a multiple of tournament size')
        return
    
    tmp_population = population.copy()
    np.random.shuffle(tmp_population)
    # print(tmp_population)
    selected_chromosome = []
    number_of_group = int(len(population) / tournament_size)
    for i in range(number_of_group):
        candidates = [tmp_population[i*4+k] for k in range(0, 4)]
        best_candidate = Fight(candidates, type_function, k)
        selected_chromosome.append(best_candidate)

    return selected_chromosome

def Fight(candidates, type_function, k=-9999):
    scores = [Evaluate(c, type_function, k) for c in candidates]
    # print(scores)
    index_best_candidate = scores.index(max(scores))
    best_candidate = candidates[index_best_candidate]

    return best_candidate

def Terminate(population):
    return np.all(population == population[0])

def Run(population_size, parameters_size, tour_size, type_function, type_cross, k=-9999):
    population = Initialize(population_size, parameters_size)
    # number_iteration = 0
    while not Terminate(population):
        # print(population)
        offspring = Crossover(population, l, type_cross)
        popop = Popop(population, offspring)
        population = np.concatenate([TournamentSelection(popop, tour_size, type_function, k), \
            TournamentSelection(popop, tour_size, type_function, k)])
        # number_iteration += 1
        # print(number_iteration)
        # print(population)

    # print(population)
    # print(number_iteration)
    return population

def is_Success(population):
    one_array = [1] * len(population[0])
    return np.all(population == one_array)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--number of parameters", required=True, type=int, help="number of parameters")
    ap.add_argument("-k", "--trap k function", type=int, default=-9999, help="Trap k function")
    ap.add_argument("-t", "--tournament size", required=True, type=int, help="Tournament's size")
    ap.add_argument("-f", "--function type", required=True, help="Type of function")
    ap.add_argument("-c", "--crossover type", required=True, help="Type of crossover")
    args = vars(ap.parse_args())
    l = args["number of parameters"]
    k = args["trap k function"]
    tour_size = args["tournament size"]
    type_func = args["function type"]
    type_cross = args["crossover type"]
    if type_func == 'trap5' and k is None:
        ap.error("k argument is required when type_func is trap5")
        

    f = open('result_ver2.txt', 'a')
    f.write('\n- l = {}, tournament size: {}, type function: {}, type crossover: {}\n'.format(l, tour_size, type_func, type_cross))
    
    seed_groups = [(18520176+i) for i in range(0,100,10)]
    MRPSs = []
    Evaluations = []
    for bisection, group in zip(range(10), seed_groups):
        print('{}-th bisection'.format(bisection + 1))
        f.write("\t{}-th bisection\n".format(bisection + 1))
        # print(group)

        global number_of_evaluations
        number_of_evaluations = 0

        num_fails = 0

        # Giai đoạn 1: Tìm cận trên
        N_upper = 4
        exceed = False
        while True:
            past_all_test = True
            for i in range(group+0, group+10):
                np.random.seed(i)
                result = Run(N_upper, l, tour_size, type_func, type_cross, k)
                if not is_Success(result):
                    past_all_test = False
                    break
            if past_all_test:
                # print('N_upper = {} past 10 tests'.format(N_upper))
                print('N_upper after stage 1: {}'.format(N_upper))
                f.write("\t\t\tN_upper after stage 1: {}\n".format(N_upper))
                break
            N_upper = N_upper * 2
            if N_upper > 8192:
                exceed = True
                num_fails += 1
                print('N_upper exceed 8192')
                f.write("\t\t\tN_upper exceed 8192\n")
                break
                
        # Giai đoạn 2: Tìm giá trị của MRPS
        if exceed == True:
            continue
        else:
            N_lower = N_upper / 2
            while (N_upper - N_lower) / N_upper > 0.1:
                N = int((N_upper + N_lower) / 2)
                past_all_test = True
                for i in range(group+0, group+10):
                    np.random.seed(i)
                    result = Run(N, l, tour_size, type_func, type_cross, k)
                    if not is_Success(result):
                        past_all_test = False
                        break
                
                if past_all_test == True:
                    N_upper = N
                else:
                    N_lower = N
                if N_upper - N_lower <= 2:
                    break
        
        MRPSs.append(N_upper)
        average_number_evaluations = number_of_evaluations / (10 - num_fails)
        Evaluations.append(average_number_evaluations)
        print('MRPS: {}'.format(N_upper))
        f.write("\t\t\tMRPS: {}\n".format(N_upper))
        print('Average number of evaluations: {}'.format(average_number_evaluations))
        f.write("\t\t\tAverage number of evaluations: {}\n".format(average_number_evaluations))

    if len(MRPSs) != 0:
        mean_MRPS = np.mean(MRPSs).round(2)
        print('Mean MRPS: {}'.format(mean_MRPS))
        f.write('\tMean MRPS: {}\n'.format(mean_MRPS))

        std_MRPS = np.std(MRPSs).round(2)
        print('std MRPS: {}'.format(std_MRPS))
        f.write('\tstd MRPS: {}\n'.format(std_MRPS))

        mean_eval = np.mean(Evaluations).round(2)
        print('Mean number of evalution: {}'.format(mean_eval))
        f.write('\tMean number of evalution: {}\n'.format(mean_eval))

        std_eval = np.std(Evaluations).round(2)
        print('std number of evaluations: {}'.format(std_eval))
        f.write('\tstd number of evaluations: {}\n'.format(std_eval))

    f.close()