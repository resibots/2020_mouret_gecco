## Abstract 

Quality Diversity (QD) algorithms are a recent family of optimization algorithms that search for a large set of diverse but high-performing solutions. Interestingly, they can solve multiple tasks at once. For instance, they can find the joint positions required for a robotic arm to reach a set of points, which can also be solved by running a classic optimizer for each target point. However, they cannot solve multiple tasks when the fitness needs to be evaluated independently for each task (e.g., optimizing policies to grasp many different objects). In this paper, we propose an extension of the MAP-Elites algorithm, called Multi-task MAP-Elites, that solves multiple tasks when the fitness function depends on the task. We evaluate it on a simulated parametrized planar arm (10-dimensional search space; 5000 tasks) and on a 6-legged robot with legs of different lengths (36-dimensional search space; 2000 tasks). The results show that in both cases our algorithm outperforms the optimization of each task separately with the CMA-ES algorithm.


______

## Introduction


Quality Diversity (QD) algorithms are a recent family of optimization algorithms that search for a large set of diverse but high-performing solutions <dt-cite key="mouret2015illuminating,cully2017quality,pugh2016quality"></dt-cite>, instead of the global optimum, like in single-objective optimization, or the Pareto frontier, like in multi-objective optimization. For instance, when optimizing aerodynamic 3D shapes, a user might want to be presented with multiple low-drag solutions of diverse materials and curvatures, and then select the best one according to criteria that are not encoded in the fitness function, such as aesthetics <dt-cite key="gaier2018"></dt-cite>
paces, and to find effective representations for future uses.

<div style="text-align: center;">
<img class="b-lazy" src="assets/png/intro.png" style="width: 75%;"/>
<br/>
<figcaption style="text-align: left;">
<b>Data-Driven Encoding MAP-Elites (DDE-Elites) searches the space of representations to search for solutions</b><br/>
 A data-driven encoding (DDE) is learned by training a VAE on the MAP-Elites archive. High fitness solutions, which increase the bias of the DDE toward performance, are found using the DDE. Novel solutions, which increase the range of solutions which can be expressed, are be found using mutation operators. UCB1, a bandit algorithm, balances the mix of these explorative and exploitative operators.
</figcaption>
</div>



______

## Related Work

*If you would like to discuss any issues or give feedback, please visit the [GitHub](https://github.com/weightagnostic/weightagnostic.github.io/issues) repository of this page for more information.*
