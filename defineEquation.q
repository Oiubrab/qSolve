 / building the solver

\l sigmoid.q

equate:{[weights;biases;inputs]
    / needs to be at lest one level of weights and biases
    firstCombination:{(last x) + x[-1_til count x] mmu y}[;inputs] each (weights[0],'biases[0]);
    firstLevel:sigmoid each firstCombination
 }
