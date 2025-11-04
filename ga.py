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
        # Each schedule assigns one program to each time slot (allow repeats)
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
    """Single-point crossover."""
    point = random.randint(1, len(parent1) - 2)
    child = parent1[:point] + parent2[point:]
    return child


def mutate(schedule, mut_rate, programs):
    """Mutate a schedule by replacing programs randomly."""
    for i in range(len(schedule)):
        if random.random() < mut_rate:
            schedule[i] = random.choice(programs)
    return schedule


def genetic_algorithm(programs, ratings, pop_size, co_r, mut_r, generations=100):
    """Run the Genetic Algorithm to optimize scheduling."""
    time_slots = ratings.columns
    population = initialize_population(programs, time_slots, pop_size)
    best_schedule = None
    best_fitness = -1

    for _ in range(generations):
        selected = selection(population, ratings, time_slots)
        next_gen = []

        for _ in range(len(population)):
            p1 = random.choice(selected)
            p2 = random.choice(selected)
            if random.random() < co_r:
                child = crossover(p1, p2)
            else:
                child = p1.copy()
            child = mutate(child, mut_r, programs)
            next_gen.append(child)

        population = next_gen
        fittest = max(population, key=lambda s: fitness(s, ratings, time_slots))
        f = fitness(fittest, ratings, time_slots)

        if f > best_fitness:
            best_fitness = f
            best_schedule = fittest

    return best_schedule, best_fitness


# --------------------------------------------------------
# Streamlit Interface
# --------------------------------------------------------

st.title("📺 Program Scheduling Optimization using Genetic Algorithm")
st.markdown("""
This Streamlit app reads your **program_ratings_modify.csv** file, detects all available programs and time slots, 
and applies a **Genetic Algorithm (GA)** to find the best schedule that maximizes audience ratings.
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
        generations = st.sidebar.number_input("Generations", 10, 500, 100)

        # Trials Section
        st.subheader("⚙️ Run Three Trials with Different Parameters")

        trial_results = []
        for i in range(1, 4):
            st.markdown(f"### 🧩 Trial {i}")
            
            # Sliders following assignment rules
            co_r = st.slider(
                f"Crossover Rate (Trial {i})",
                min_value=0.0,
                max_value=0.95,
                value=0.8,     # Default as required
                step=0.01,
                key=f"co_{i}"
            )

            mut_r = st.slider(
                f"Mutation Rate (Trial {i})",
                min_value=0.01,
                max_value=0.05,
                value=0.02,    # Within allowed range
                step=0.01,
                key=f"mut_{i}"
            )

            run = st.button(f"Run Trial {i}", key=f"run_{i}")

            if run:
                schedule, score = genetic_algorithm(programs, rating_matrix, pop_size, co_r, mut_r, generations)

                # Display the resulting schedule
                result_df = pd.DataFrame({
                    "Time Slot": time_slots,
                    "Program": schedule
                })

                st.write(f"**Trial {i} Parameters:** CO_R = {co_r}, MUT_R = {mut_r}")
                st.table(result_df)
                st.write(f"**Total Fitness (Sum of Ratings): {score:.2f}**")

                trial_results.append((i, co_r, mut_r, score, schedule))

        # Display summary of all trials
        if trial_results:
            st.subheader("📊 Summary of All Trials")
            summary_df = pd.DataFrame(
                trial_results,
                columns=["Trial", "CO_R", "MUT_R", "Fitness", "Best Schedule"]
            )
            st.dataframe(summary_df)

else:
    st.error("❌ Could not find 'program_ratings_modify.csv'. Please make sure it’s in the same folder as this app.")



