# ==========================================
# Streamlit App: Scheduling using Genetic Algorithm (3 Trials)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import random

# ------------------------------------------
# 1. Load Dataset
# ------------------------------------------
st.title("📺 TV Program Scheduling using Genetic Algorithm")

st.markdown("""
This application uses a *Genetic Algorithm (GA)* to schedule TV programs based on their ratings.  
You can adjust the *Crossover Rate (CO_R)* and *Mutation Rate (MUT_R)* to observe how they affect each generated schedule.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("program_ratings_modify.csv")
    return df

try:
    df = load_data()
    st.subheader("📊 Program Rating Data (From CSV)")
    st.dataframe(df)

    programs = df["Type of Program"].tolist()
    hours = df.columns[1:]
except Exception as e:
    st.error("❌ Could not load 'program_ratings_modify.csv'. Please make sure it exists.")
    st.stop()

# ------------------------------------------
# 2. GA Parameters (fixed for all trials)
# ------------------------------------------
POP_SIZE = 20
GENERATIONS = 50

# ------------------------------------------
# 3. Genetic Algorithm Functions
# ------------------------------------------
def fitness(schedule, df):
    """Calculate total rating for a given schedule."""
    total = 0
    for h, prog in enumerate(schedule):
        total += df.loc[df["Type of Program"] == prog, df.columns[h + 1]].values[0]
    return total

def roulette_selection(population, fitnesses):
    """Roulette-wheel selection."""
    total_fit = sum(fitnesses)
    if total_fit == 0:
        return random.choice(population)
    probs = [f / total_fit for f in fitnesses]
    return population[np.random.choice(len(population), p=probs)]

def crossover(parent1, parent2, rate):
    """Single-point crossover."""
    if random.random() < rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

def mutate(schedule, rate):
    """Randomly mutate some genes."""
    for i in range(len(schedule)):
        if random.random() < rate:
            schedule[i] = random.choice(programs)
    return schedule

def genetic_algorithm(df, CO_R, MUT_R, pop_size=POP_SIZE, generations=GENERATIONS):
    """Run the genetic algorithm."""
    population = [random.choices(programs, k=len(hours)) for _ in range(pop_size)]
    best_individual, best_fitness = None, -1

    for _ in range(generations):
        fitnesses = [fitness(ind, df) for ind in population]
        new_population = []

        # --- Elitism ---
        elite = population[np.argmax(fitnesses)]
        elite_fit = max(fitnesses)
        if elite_fit > best_fitness:
            best_fitness = elite_fit
            best_individual = elite
        new_population.append(elite)

        # --- Create new population ---
        while len(new_population) < pop_size:
            p1 = roulette_selection(population, fitnesses)
            p2 = roulette_selection(population, fitnesses)
            c1, c2 = crossover(p1, p2, CO_R)
            c1 = mutate(c1, MUT_R)
            c2 = mutate(c2, MUT_R)
            new_population += [c1, c2]

        population = new_population[:pop_size]

    return best_individual, best_fitness

# ------------------------------------------
# 4. Run 3 Trials Section
# ------------------------------------------
st.subheader("⚙️ Run Three Trials with Different Parameters")
trial_results = []

# ===== Trial 1 =====
st.markdown("## 🧩 Trial 1")
CO_R1 = st.slider("Crossover Rate (Trial 1)", 0.0, 0.95, 0.8, 0.01, key="co1")
MUT_R1 = st.slider("Mutation Rate (Trial 1)", 0.01, 0.20, 0.20, 0.01, key="mut1")
if st.button("Run Trial 1"):
    best1, fit1 = genetic_algorithm(df, CO_R1, MUT_R1)
    df1 = pd.DataFrame({"Hour": hours, "Program": best1})
    st.success(f"✅ Trial 1 done — Total Fitness: {fit1:.2f}")
    st.dataframe(df1)
    trial_results.append((1, CO_R1, MUT_R1, fit1, best1))

# ===== Trial 2 =====
st.markdown("## 🧩 Trial 2")
CO_R2 = st.slider("Crossover Rate (Trial 2)", 0.0, 0.95, 0.9, 0.01, key="co2")
MUT_R2 = st.slider("Mutation Rate (Trial 2)", 0.01, 0.20, 0.03, 0.01, key="mut2")
if st.button("Run Trial 2"):
    best2, fit2 = genetic_algorithm(df, CO_R2, MUT_R2)
    df2 = pd.DataFrame({"Hour": hours, "Program": best2})
    st.success(f"✅ Trial 2 done — Total Fitness: {fit2:.2f}")
    st.dataframe(df2)
    trial_results.append((2, CO_R2, MUT_R2, fit2, best2))

# ===== Trial 3 =====
st.markdown("## 🧩 Trial 3")
CO_R3 = st.slider("Crossover Rate (Trial 3)", 0.0, 0.95, 0.6, 0.01, key="co3")
MUT_R3 = st.slider("Mutation Rate (Trial 3)", 0.01, 0.20, 0.01, 0.01, key="mut3")
if st.button("Run Trial 3"):
    best3, fit3 = genetic_algorithm(df, CO_R3, MUT_R3)
    df3 = pd.DataFrame({"Hour": hours, "Program": best3})
    st.success(f"✅ Trial 3 done — Total Fitness: {fit3:.2f}")
    st.dataframe(df3)
    trial_results.append((3, CO_R3, MUT_R3, fit3, best3))

# ===== Summary =====
if len(trial_results) > 0:
    st.markdown("## 📊 Summary of All Trials")
    summary = pd.DataFrame(trial_results, columns=["Trial", "CO_R", "MUT_R", "Fitness", "Best Schedule"])
    st.dataframe(summary)

    best_trial = max(trial_results, key=lambda x: x[3])
    st.success(f"🏆 Best Performing Trial: Trial {best_trial[0]} (Fitness: {best_trial[3]:.2f})")

