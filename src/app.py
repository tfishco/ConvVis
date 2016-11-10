from flask import Flask, render_template, request, send_file

###############################################################################
#################################### GA #######################################
###############################################################################

import sys
import random
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatches
import numpy as np

class Individual:
    def __init__(self, gene):
        self.gene = gene
        self.fitness = 0

    def get_gene(self):
        return self.gene

    def set_gene(self, gene):
        self.gene = gene

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_string(self):
        s = ""
        for i in range(0, len(self.gene)):
            s = s + str(self.gene[i])
        return s + " | " + str(self.fitness)


def GAMain(gene_size, population_size, generations, mutation_rate):
    population = []
    new_population = []

    mean_fitnesses = []
    best_fitnesses = []

    initalise(population_size, gene_size, population)
    evaluate(population_size, gene_size, population)
    for i in range(0, generations):
        best_individual = get_best_individual(population_size, population)
        select(population_size, gene_size, population, new_population)
        xover(population_size, gene_size, new_population)
        mutate(population_size, gene_size, new_population, mutation_rate)
        evaluate(population_size, gene_size, new_population)
        new_population[get_worst_index(population_size, population)] = best_individual

        total_fitness = get_total_fitness(population_size, new_population)
        mean_fitness = get_mean_fitness(population_size,gene_size, total_fitness)
        population = list(new_population)
        new_population = []
        mean_fitnesses.append(mean_fitness)
        best_fitnesses.append(best_individual.get_fitness())
    return list([mean_fitnesses,best_fitnesses])


def initalise(population_size, gene_size, population):
    for i in range(0, population_size):
        population.append(Individual(random_gene(gene_size)))

def evaluate(population_size, gene_size, population):
    for i in range(0, population_size):
        count = 0
        for j in range(0, gene_size):
            if population[i].get_gene()[j] == 1:
                count += 1
        population[i].set_fitness(count)

def select(population_size, gene_size, population, new_population):
    for i in range(0, population_size):
        parent1 = random.randint(0, population_size - 1)
        parent2 = random.randint(0, population_size - 1)
        if population[parent1].get_fitness() > population[parent2].get_fitness():
            parent_gene = population[parent1].get_gene()
        else:
            parent_gene = population[parent2].get_gene()
        child = Individual(parent_gene)
        new_population.append(child)

def xover(population_size, gene_size, population):
    for i in range(0, population_size, 2):
        rand = random.randint(1, gene_size - 1)
        for j in range(rand, gene_size):
            population[i].get_gene()[j], population[i + 1].get_gene()[j] = population[i + 1].get_gene()[j], population[i].get_gene()[j]

def mutate(population_size, gene_size, population, mutation_rate):
    for i in range(0, population_size):
        temp_gene = population[i].get_gene()[:]
        for j in range(0, gene_size):
            rand = random.random()
            if rand < mutation_rate:
                temp_gene[j] = temp_gene[j] ^ 1
        population[i].set_gene(temp_gene)

def get_total_fitness(population_size, population):
    count = 0
    for i in range(0, population_size):
        count += population[i].get_fitness()
    return count

def get_mean_fitness(population_size, gene_size, total_fitness):
    return total_fitness / population_size

def get_best_fitness(population_size, population):
    best = 0
    for i in range(0, population_size):
        if population[i].get_fitness() > best:
            best = population[i].get_fitness()
    return best

def get_best_individual(population_size, population):
    ind = None
    fitness = 0
    for i in range(0,population_size):
        if population[i].get_fitness() > fitness:
            ind = population[i]
            fitness = population[i].get_fitness()
    return ind

def get_worst_index(population_size, population):
    worst_fitness = 50
    worst_index = -1
    for i in range(0, population_size):
        if population[i].get_fitness() < worst_fitness:
            worst_fitness = population[i].get_fitness()
            worst_index = i
    return worst_index

def random_gene(gene_size):
    gene = [None] * gene_size
    for i in range(0, gene_size):
        gene[i] = random.randint(0,1)
    return gene

def print_population(population):
    for i in range(0 , len(population)):
        print(population[i].get_string())

def print_gene(gene_size, gene):
    s = ""
    for i in range(0, gene_size):
        s += str(gene[i]) + " "
    print(s)


###############################################################################
################################## Web App ####################################
###############################################################################
app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Flask root</h1>"

@app.route("/ga")
def ga():
    if request.args.get("gene_size")==None:
        gene_size=26
    else:
        gene_size = int(request.args.get("gene_size"))

    if request.args.get("population_size")==None:
        population_size=26
    else:
        population_size = int(request.args.get("population_size"))

    if request.args.get("generations")==None:
        generations=26
    else:
        generations = int(request.args.get("generations"))

    if request.args.get("mutation_rate")==None:
        mutation_rate=0.1
    else:
        mutation_rate = float(request.args.get("mutation_rate"))

    fitnessData = GAMain(gene_size, population_size, generations, mutation_rate)
    max = 100
    return render_template("main.html",fitnessData=fitnessData,max=max)

if __name__ == "__main__":
    app.debug = True
    app.run()
