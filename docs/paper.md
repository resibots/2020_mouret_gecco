## Abstract 

The way solutions are represented, or encoded, is usually the result of domain knowledge and experience. In this work, we combine MAP-Elites with Variational Autoencoders to learn a Data-Driven Encoding (DDE) that captures the essence of the highest-performing solutions while still able to encode a wide array of solutions. Our approach learns this data-driven encoding during optimization by balancing between exploiting the DDE to generalize the knowledge contained in the current archive of elites and exploring new representations that are not yet captured by the DDE. Learning representation during optimization allows the algorithm to solve high-dimensional problems, and provides a low-dimensional representation which can be then be re-used. We evaluate the DDE approach by evolving solutions for inverse kinematics of a planar arm (200 joint angles) and for gaits of a 6-legged robot in action space (a sequence of 60 positions for each of the 12 joints). We show that the DDE approach not only accelerates and improves optimization, but produces a powerful encoding that captures a bias for high performance while expressing a variety of solutions.
  

______

## Introduction

How solutions are represented is one of the most critical design decisions in optimization, as the representation defines the way an algorithm can move in the search space<dt-cite key="rothlauf2006representations"></dt-cite>. Work on representations tends to focus on encoding priors or innate biases: aerodynamic designs evolved with splines to encourage smooth forms <dt-cite key="olhofer2001adaptive"></dt-cite>, Compositional Pattern Producing Networks (CPPNs) introduce biases for symmetry and repetition to produce images and neural network weight patterns <dt-cite key="cppn,hyperneat"></dt-cite>, or encodings which aim to encourage modularity in neural networks<dt-cite key="mouret2008mennag,durr2010genetic,doncieux2004evolving"></dt-cite>.  

The best representations balance a bias for high performing solutions, so they can easily be discovered, and the ability to express a diversity of potential solutions, so the the search space can be widely explored. At the one extreme, a representation which only encodes the global optimum is easy to search, but useless for finding any other solution. At the other, a representation which can encode anything presents a difficult and dauntingly vast search space.

Given a large set of example solutions, representations could be learned from data instead of been hand-tailored by trial-and-error: a learned representation would replicate the same biases toward performance and the same range of expressivity as the source data set. For instance, given a dataset of face images, a variational autoencoder (VAE) <dt-cite key="vae"></dt-cite> or a Generative Adversarial Network (GAN) <dt-cite key="gan"></dt-cite> can learn a low-dimensional latent space, that is, a representation, that makes it possible to explore the space of face images. In essence, the decoder which maps the latent space to the phenotypic space learns the "recipe" of faces. Importantly, the existence of such a low-dimensional latent space is possible because _the dataset is a very small part of the set of all possible images_.

However, using a dataset of preselected high-performing solutions "traps" the search within the distribution of solutions that are already known: a VAE trained on white faces will never generate a black face. This limits the usefulness of such data-driven representations for discovering _novel_ solutions to hard problems.

In this paper, we propose the use of the MAP-Elites algorithm <dt-cite key="mapelites"></dt-cite> to automatically generate a dataset for representations using only a performance function and a diversity space. Quality diversity algorithms like MAP-Elites are a good fit for representation discovery: creating archives of diverse high-performing solutions is precisey their purpose. Using the MAP-Elites archive as a source of example solutions, we can capture the genetic distribution of the highest performing solutions, or elites, by training a VAE and obtaining a latent representation. As the VAE is only trained on elites, this learned representation, or data-driven encoding (DDE), has a strong bias towards solutions with high fitness; and because the elites having varying phenotypes, the DDE is able to express a range of solutions. Though the elites vary along a phenotypic continuum, they commonly have many genotypic similarities<dt-cite key="me_linemut"></dt-cite>, which makes it likely to find a good latent space.

Nonetheless, MAP-Elites will struggle to find high-performing solutions without an adequate representation. Fortunately, the archive is produced by MAP-Elites in an iterative, any-time fashion, so there is no "end state" to wait for before a DDE can be trained -- a DDE can be trained _during optimization_. The DDE can then be used to enhance the optimization process. By improving the quality of the archive, the DDE imporves the quality of its own source data, establishing a virtuous cycle of archive and encoding improvement.

A DDE based on an archive will encounter the same difficulty as any learned encoding: the DDE can only represent solutions that are already in the dataset. How then, can we discover new solutions? Fundamentally, to search for an encoding, we need to both _exploit the best known representation_, that is, create better solutions according to the current best "recipes", and also _explore new representations_ -- solutions which do not follow any "recipe".


In this paper, we address this challenge by mixing solutions generated with the DDE with solutions obtained using standard MAP-Elites operators. Our algorithm applies classic operators, such as Gaussian mutation, to create candidates which could not be captured by the current DDE. At the same time we leverage the DDE to generalize common patterns across the map and create new solutions that are likely to be high-performing. To avoid introducing new hyper-parameters, we tune this exploration/exploitation trade-off optimally using a multi-armed bandit algorithm <dt-cite key="garivier2011upper"></dt-cite>.

This new algorithm, DDE-Elites, reframes optimization as a search for representations. Integrating MAP-Elites with a VAE makes it possible to apply  quality diversity to high-dimensional search spaces, and to find effective representations for future uses.

<div style="text-align: center;">
<img class="b-lazy" src="assets/png/intro.png" style="width: 75%;"/>
<br/>
<figcaption style="text-align: left;">
<b>Data-Driven Encoding MAP-Elites (DDE-Elites) searches the space of representations to search for solutions</b><br/>
 A data-driven encoding (DDE) is learned by training a VAE on the MAP-Elites archive. High fitness solutions, which increase the bias of the DDE toward performance, are found using the DDE. Novel solutions, which increase the range of solutions which can be expressed, are be found using mutation operators. UCB1, a bandit algorithm, balances the mix of these explorative and exploitative operators.
</figcaption>
</div>

We are interested in domains that have a straightforward low-level representation that is too high-dimensional for most algorithms, for instance:  joints positions at every time-step for a walking robot (12 $\times$ 60=720 positions for a 3-second gait of a robot with 12 degrees of freedom), 3D shapes in which each voxel is encoded individually (low-level representation would be 1000-dimensional for a 10 $\times$ 10 $\times$ 10 grid), images encoded in the pixel-space, etc.

Ideally, the generated DDE will capture the main regularities of the domain. In robot locomotion, this could correspond to periodic functions, since we already know that a $36$-dimensional controller based on periodic functions can produce the numerous joint commands required every second to effectively drive a 12-joint walking robot in many different ways<dt-cite key=cully2015robots></dt-cite>. In many domains the space of possible solutions can be vast, while the inherent dimensionality of interesting solutions still compact. By purposefully seeking out a space of solutions, rather than individual solutions themselves, we can solve high-dimensional problems in a lower dimensional space.


______

## Related Work

*If you would like to discuss any issues or give feedback, please visit the [GitHub](https://github.com/weightagnostic/weightagnostic.github.io/issues) repository of this page for more information.*
