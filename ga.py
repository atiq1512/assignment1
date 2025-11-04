import streamlit as st
import pandas as pd
import random
import os

# --------------------------------------------------------
# Genetic Algorithm Helper Functions
# --------------------------------------------------------

def initialize_population(programs, time_slots, pop_size):
    """Create initial random schedules that match number of time slots."""
    population = []
    for _ in range(pop_size):
        schedule = [random.choice(programs) for _ in time_slots]
        population.append(schedule)
    return population


def fitness(schedule, ratings, time_slots):
    """Compute total rating score of a schedule based on program-hour match."""
    total = 0
    for program, slot in zip(schedule, time_slots):
        total += ratings.loc[program, slot]
    return total


def selection(population, ratings, time_slots):
    """Select the top 50% of schedules with best fitness."""
    sorted_pop = sorted(population, key=lambda s: fitness(s, ratings, time_slots), reverse=True)
    return sorted_pop[:len(population)//2]


def crossover(parent1, parent2):
    """Single-point crossover to combine two parent schedules."""
    if len(parent1) < 2:
        return parent1
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child


def mutate(schedule, mut_rate, programs):
    """Mutate a schedule by replacing programs randomly."""
    for i in range(len(schedule)):
        if random.random() < mut_rate:
            schedule[i] = random.choice(programs)
    return schedule


def genetic_algorithm(programs, ratings, pop_size, co_r, mut_r, generations=300):
    """Run the Genetic Algorithm to optimize scheduling."""
    time_slots = ratings.columns
    population = initialize_population(programs, time_slots, pop_size)
    best_schedule = None
    best_fitness = -1

    for gen in range(generations):
        # Evaluate current population
        scored_pop = [(chrom, fitness(chrom, ratings, time_slots)) for chrom in population]
        scored_pop.sort(key=lambda x: x[1], reverse=True)

        # Keep best individual (elitism)
        elite = scored_pop[0][0]
        elite_score = scored_pop[0][1]

        if elite_score > best_fitness:
            best_schedule = elite
            best_fitness = elite_score

        # Selection
        selected = [chrom for chrom, _ in scored_pop[:len(scored_pop)//2]]

        # Generate next generation
        next_gen = [elite]  # preserve the best one
        while len(next_gen) < pop_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            if random.random() < co_r:
                child = crossover(parent1, parent2)
            else:
                child = parent1.copy()
            child = mutate(child, mut_r, programs)
            next_gen.append(child)

        population = next_gen

    return best_schedule, best_fitness


# --------------------------------------------------------
# Streamlit Interface
# --------------------------------------------------------

st.title("📺 Program Scheduling Optimization using Genetic Algorithm (Improved Version)")
st.markdown("""
This Streamlit app reads your **program_ratings_modify.csv** file, detects all available programs and time slots, 
and applies a **Genetic Algorithm (GA)** to find the best schedule that maximizes audience ratings.

Now improved with:
- ✅ Elitism (keeps best schedule per generation)
- ✅ Visible fitness differences
- ✅ True parameter sensitivity (CO_R, MUT_R)
""")

csv_path = "program_ratings_modify.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    if "Type of Program" not in df.columns:
        st.error("❌ The CSV must contain a column named 'Type of Program'.")
    else:
        st.success("✅ CSV file loaded successfully!")
        st.write(f"**Detected {len(df)} programs and {len(df.columns) - 1} time slots.**")
        st.dataframe(df)

        # Extract programs and ratings
        programs = df["Type of Program"].tolist()
        rating_matrix = df.set_index("Type of Program")
        time_slots = rating_matrix.columns.tolist()

        # Sidebar for GA settings
        st.sidebar.header("Genetic Algorithm Parameters")
        pop_size = st.sidebar.number_input("Population Size", 100, 1000, 300)
        generations = st.sidebar.number_input("Generations", 50, 1000, 300)

        # Trials Section
        st.subheader("⚙️ Run Three Trials with Different Parameters")

        trial_results = []  # Store results for summary

        # --------- TRIAL 1 ----------
        st.markdown("## 🧩 Trial 1")
        co_r1 = st.slider("Crossover Rate (Trial 1)", 0.0, 0.95, 0.8, 0.01, key="co1")
        mut_r1 = st.slider("Mutation Rate (Trial 1)", 0.01, 0.05, 0.02, 0.01, key="mut1")

        if st.button("Run Trial 1"):
            schedule1, score1 = genetic_algorithm(programs, rating_matrix, pop_size, co_r1, mut_r1, generations)
            result_df1 = pd.DataFrame({"Time Slot": time_slots, "Program": schedule1})
            st.success(f"✅ Trial 1 Completed - CO_R={co_r1}, MUT_R={mut_r1}")
            st.table(result_df1)
            st.write(f"**Total Fitness (Sum of Ratings): {score1:.5f}**")
            trial_results.append((1, co_r1, mut_r1, score1, schedule1))

        # --------- TRIAL 2 ----------
        st.markdown("## 🧩 Trial 2")
        co_r2 = st.slider("Crossover Rate (Trial 2)", 0.0, 0.95, 0.9, 0.01, key="co2")
        mut_r2 = st.slider("Mutation Rate (Trial 2)", 0.01, 0.05, 0.03, 0.01, key="mut2")

        if st.button("Run Trial 2"):
            schedule2, score2 = genetic_algorithm(programs, rating_matrix, pop_size, co_r2, mut_r2, generations)
            result_df2 = pd.DataFrame({"Time Slot": time_slots, "Program": schedule2})
            st.success(f"✅ Trial 2 Completed - CO_R={co_r2}, MUT_R={mut_r2}")
            st.table(result_df2)
            st.write(f"**Total Fitness (Sum of Ratings): {score2:.5f}**")
            trial_results.append((2, co_r2, mut_r2, score2, schedule2))

        # --------- TRIAL 3 ----------
        st.markdown("## 🧩 Trial 3")
        co_r3 = st.slider("Crossover Rate (Trial 3)", 0.0, 0.95, 0.6, 0.01, key="co3")
        mut_r3 = st.slider("Mutation Rate (Trial 3)", 0.01, 0.05, 0.01, 0.01, key="mut3")

        if st.button("Run Trial 3"):
            schedule3, score3 = genetic_algorithm(programs, rating_matrix, pop_size, co_r3, mut_r3, generations)
            result_df3 = pd.DataFrame({"Time Slot": time_slots, "Program": schedule3})
            st.success(f"✅ Trial 3 Completed - CO_R={co_r3}, MUT_R={mut_r3}")
            st.table(result_df3)
            st.write(f"**Total Fitness (Sum of Ratings): {score3:.5f}**")
            trial_results.append((3, co_r3, mut_r3, score3, schedule3))

        # --------- SUMMARY ----------
        if len(trial_results) > 0:
            st.markdown("## 📊 Summary of All Trials")
            summary_df = pd.DataFrame(
                trial_results,
                columns=["Trial", "CO_R", "MUT_R", "Fitness", "Best Schedule"]
            )
            st.dataframe(summary_df)

else:
    st.error("❌ Could not find 'program_ratings_modify.csv'. Please make sure it’s in the same folder as this app.")


