import streamlit as st
import pandas as pd
import random
import os

# --------------------------------------------------------
# Genetic Algorithm Helper Functions
# --------------------------------------------------------

def initialize_population(programs, pop_size):
    population = []
    for _ in range(pop_size):
        schedule = random.sample(programs, len(programs))
        population.append(schedule)
    return population

def fitness(schedule, ratings):
    total = 0
    for program in schedule:
        total += ratings.get(program, 0)
    return total

def selection(population, ratings):
    sorted_pop = sorted(population, key=lambda s: fitness(s, ratings), reverse=True)
    return sorted_pop[:len(population)//2]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-2)
    child = parent1[:point] + [p for p in parent2 if p not in parent1[:point]]
    return child

def mutate(schedule, mut_rate):
    for i in range(len(schedule)):
        if random.random() < mut_rate:
            j = random.randint(0, len(schedule)-1)
            schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule

def genetic_algorithm(programs, ratings, pop_size, co_r, mut_r, generations=100):
    population = initialize_population(programs, pop_size)
    best_schedule = None
    best_fitness = 0

    for _ in range(generations):
        selected = selection(population, ratings)
        next_gen = []
        for _ in range(len(population)):
            p1 = random.choice(selected)
            p2 = random.choice(selected)
            child = crossover(p1, p2) if random.random() < co_r else p1.copy()
            child = mutate(child, mut_r)
            next_gen.append(child)

        population = next_gen
        fittest = max(population, key=lambda s: fitness(s, ratings))
        f = fitness(fittest, ratings)
        if f > best_fitness:
            best_fitness = f
            best_schedule = fittest

    return best_schedule, best_fitness


# --------------------------------------------------------
# Streamlit Interface
# --------------------------------------------------------

st.title("Scheduling Using Genetic Algorithm")
st.markdown("""
This application demonstrates how **Genetic Algorithms** can optimize a scheduling problem
based on program ratings from your modified CSV file.
""")

# Load CSV file automatically
csv_path = "program_ratings_modify.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if "Program" not in df.columns or "Rating" not in df.columns:
        st.error("❌ The CSV file must contain 'Program' and 'Rating' columns.")
    else:
        st.success("✅ CSV file loaded successfully!")
        st.dataframe(df)

        programs = df["Program"].tolist()
        ratings = dict(zip(df["Program"], df["Rating"]))

        # Sidebar parameters
        st.sidebar.header("Algorithm Parameters")
        pop_size = st.sidebar.number_input("Population Size", 100, 1000, 300)
        generations = st.sidebar.number_input("Generations", 10, 500, 100)

        # Main trial section
        st.subheader("Run Three Trials with Different Parameters")

        trial_results = []
        for i in range(1, 4):
            st.markdown(f"### Trial {i}")
            co_r = st.slider(f"Crossover Rate (Trial {i})", 0.0, 0.95, 0.8, 0.01, key=f"co_{i}")
            mut_r = st.slider(f"Mutation Rate (Trial {i})", 0.01, 0.05, 0.02, 0.01, key=f"mut_{i}")
            run = st.button(f"Run Trial {i}", key=f"run_{i}")

            if run:
                schedule, score = genetic_algorithm(programs, ratings, pop_size, co_r, mut_r, generations)
                result_df = pd.DataFrame({
                    "Time Slot": [f"Slot {j+1}" for j in range(len(schedule))],
                    "Program": schedule
                })
                st.write(f"**Trial {i} Parameters:** CO_R = {co_r}, MUT_R = {mut_r}")
                st.table(result_df)
                st.write(f"**Total Fitness (Sum of Ratings): {score}**")
                trial_results.append((i, co_r, mut_r, score, schedule))

        if trial_results:
            st.subheader("📊 Summary of All Trials")
            summary_df = pd.DataFrame(trial_results, columns=["Trial", "CO_R", "MUT_R", "Fitness", "Best Schedule"])
            st.dataframe(summary_df)
else:
    st.error(f"❌ Could not find '{csv_path}'. Please ensure it is in the same folder as this app.")
