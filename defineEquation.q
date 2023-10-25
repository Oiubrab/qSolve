 / building the solver

{[weights;biases;inputs]
    / needs to be at lest one level of weights and biases
    firstCombination:biases[0] + weights[0] mmu inputs;
    firstLevel:sigmoid[firstCombination];
}
