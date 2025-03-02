---
layout: null
permalink: /research/TReND/
---

<html>
<head>
  <meta charset="utf-8">
  <!-- Meta tags for social media banners, these should be filled in appropriatly as they are your "business card" -->
  <!-- Replace the content tag with appropriate information -->
  <meta name="description"
    content="Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment">
  <meta property="og:title" content="AAMAS LLM-MCA" />
  <meta property="og:description"
    content="Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment" />
  <meta property="og:url" content="https://iconlab.negarmehr.com/AAMAS-LLM-MCA/" />

  <meta name="twitter:title" content="AAMAS LLM-MCA">
  <meta name="twitter:description"
    content="Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment">
  <!-- Keywords for your paper to be indexed by-->
  <meta name="keywords"
    content="Credit Assignment, Task Allocation, Multi-Agent Reinforcement Learning, Large Language Models, Foundation Models">
  <meta name="viewport" content="width=device-width, initial-scale=1">


  <title>CDC POLICEd RL</title>
  <link rel="icon" type="image/x-icon" href="static/images/favicon.ico">
  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro" rel="stylesheet">

  <link rel="stylesheet" href="static/css/bulma.min.css">
  <link rel="stylesheet" href="static/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="static/css/bulma-slider.min.css">
  <link rel="stylesheet" href="static/css/fontawesome.all.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="static/css/index.css">

  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://documentcloud.adobe.com/view-sdk/main.js"></script>
  <script defer src="static/js/fontawesome.all.min.js"></script>
  <script src="static/js/bulma-carousel.min.js"></script>
  <script src="static/js/bulma-slider.min.js"></script>
  <script src="static/js/index.js"></script>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]} });
	  MathJax.Hub.Config({ TeX: { equationNumbers: {autoNumber: "AMS"} } });
  </script>
  <script type="text/javascript" async
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
  </script>

  <style>
    .reference {
      margin-bottom: 3mm;
    }
  </style>
</head>

<body>


  <section class="hero">
    <div class="hero-body">
      <div class="container is-max-desktop">
        <div class="columns is-centered">
          <div class="column has-text-centered">
            <h1 class="title is-1 publication-title">Leveraging Large Language Models for Effective and Explainable
              Multi-Agent Credit Assignment</h1>
            <div class="is-size-5 publication-authors">
              <!-- Paper authors -->
              <span class="author-block"></span>
              <a href="https://kartik-nagpal.github.io/" target="_blank">Kartik Nagpal</a>,
              </span>
              <span class="author-block">
                <a href="http://dayiethandong.com/" target="_blank">Dayi Dong</a>,
              </span>
              <span class="author-block">
                <a href="https://jean-baptistebouvier.github.io/" target="_blank">Jean-Baptiste Bouvier</a>,
              </span>
              <span class="author-block">
                <a href="https://negarmehr.com/" target="_blank">Negar Mehr</a>
              </span><br>
              <a href="https://iconlab.negarmehr.com/" target="_blank">ICON Lab</a> at UC Berkeley <br>
              24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)
            </div>

            <div class="column has-text-centered">
              <div class="publication-links">
                <!-- Arxiv PDF link -->
                <!-- <span class="link-block">
                  <a href="https://arxiv.org/pdf/2407.20456.pdf" target="_blank"
                    class="external-link button is-normal is-rounded is-dark">
                    <span class="icon">
                      <i class="fas fa-file-pdf"></i>
                    </span>
                    <span>Paper</span>
                  </a>
                </span> -->

                <!-- Github link -->
                <span class="link-block">
                  <a href="https://github.com/labicon/LLM-MCA" target="_blank"
                    class="external-link button is-normal is-rounded is-dark">
                    <span class="icon">
                      <i class="fab fa-github"></i>
                    </span>
                    <span>Code</span>
                  </a>
                </span>

                <!-- ArXiv abstract Link -->
                <!-- <span class="link-block">
                  <a href="https://arxiv.org/abs/2407.20456" target="_blank"
                    class="external-link button is-normal is-rounded is-dark">
                    <span class="icon">
                      <i class="ai ai-arxiv"></i>
                    </span>
                    <span>arXiv</span>
                  </a>
                </span> -->
              </div>


            </div>
          </div>
        </div>
      </div>
    </div>
  </section>




  <!-- Teaser video-->
  <!-- <section class="hero teaser">
    <div class="container is-max-desktop">
      <div class="hero-body">
        <video poster="" id="tree" autoplay controls muted loop width="1265" height="843">
          <source src="static/videos/multi_trajs_shuttle_500.mp4" type="video/mp4">
        </video>
        <h4 class="subtitle has-text-centered">
          Guaranteed soft landing of the space shuttle using our learned POLICEd controller.
        </h4>
      </div>
    </div>
  </section> -->
  <!-- End teaser video -->

  <!-- Paper abstract -->
  <section class="section hero is-light">
    <div class="container is-max-desktop">
      <div class="columns is-centered has-text-centered">
        <div class="column is-four-fifths">
          <h2 class="title is-3">Abstract</h2>
          <div class="content has-text-justified">
            <p>
              Recent efforts in autonomous vehicle coordination and in-space assembly have shown the importance of
              enabling multiple robots collaboration to achieve a shared goal. A common approach for learning this
              cooperative behavior is to utilize the centralized-training decentralized-execution paradigm. However,
              this approach also introduces a new challenge: how do we evaluate the contributions of each agent's
              actions to the overall success or failure. This ``credit assignment'' problem has been extensively studied
              in the Multi-Agent Reinforcement Learning~(MARL) literature, but with little progress. In fact, humans
              performing simple inspection of the agents' trajectories often generate better credit evaluations than
              existing methods. We combine this observation with recent works which show Large Language Models~(LLMs)
              demonstrate human-level performance at many pattern recognition tasks. Our key idea is to reformulate
              credit assignment as pattern recognition problems and propose our novel Large Language Model Multi-agent
              Credit Assignment~(LLM-MCA) method. Our approach utilizes a centralized LLM reward-critic which
              numerically decomposes the overall reward based on the individualized contribution of each agent in the
              scenario. We then update the agents' policy networks based on this feedback. We also propose an extension
              (LLM-TACA) where our LLM-critic performs explicit task assignment by passing an intermediary goal directly
              to each agent in the scenario. Both our methods far outperform the state-of-the-art on a variety of
              benchmarks, including Level-Based Foraging, Robotic Warehouse, and our new ``Spaceworld'' benchmark which
              incorporates safety-related constraints. As an artifact of our methods, we generate large trajectory
              datasets with each timestep annotated with per-agent reward information, as sampled from our LLM critics.
              We hope that by making this dataset available, we will enable future works to directly train a set of
              collaborative, decentralized policies offline.
            </p>
          </div>
        </div>
      </div>
    </div>
  </section>
  <!-- End paper abstract -->






  <!-- <section class="section hero is-small">
    <div class="container">
      <div class="columns is-centered has-text-centered">
        <div class="column is-four-fifths">
          <div class="content has-text-justified">
            <p>
              Most safe RL works rely on reward shaping to discourage violations of a safety constraint.
              However, such <em>soft constraints</em> do not guarantee safety.
              Previous works trying to enforce <em>hard constraints</em> in RL typical suffer from two limitations:
              either they need an accurate model of the environment, or their learned safety certificate only
              approximate without guarantees an actual safety certificate.
            </p>
            <p>
              On the other hand, our <strong>POLICEd RL</strong> approach can provably enforce hard constraint
              satisfaction in closed-loop with a black-box environment.
              We build a repulsive buffer region in front of the constraint to prevent trajectories from approaching it.
              Since trajectories cannot cross this buffer, they also cannot violate the constraint.
            </p>
          </div>
          <img src="static/images/schema.png" alt="POLICEd RL illustration"
            style="height:300px !important; display:block !important; margin:auto !important;" />
          <div class="content has-text-justified" style="margin-top: 10px;">
            <p>
              Phase portrait of constrained output $y$ illustrating our <strong>High Relative Degree POLICEd RL</strong>
              method on a system of relative degree $2$.
              To prevent states from violating constraint $y \leq y_{max}$ (<strong><span
                  style="color:PaleVioletRed;">red dashed line</span></strong>), our policy guarantees that
              trajectories entering buffer region $\mathcal{B}$ (<strong><span
                  style="color:LightSeaGreen;">blue</span></strong>) cannot leave it through its upper bound
              (<strong><span style="color:Blue;">blue dotted line</span></strong>).
              Our policy makes $\ddot y$ sufficiently negative in buffer $\mathcal{B}$ to bring $\dot y$ to $0$ in all
              trajectories entering $\mathcal{B}$.
              Once $\dot y < 0$, trajectories cannot approach the constraint. Due to the states' inertia, it is
                physically impossible to prevent all constraint violations. For instance, $y=y_{max}$, $\dot y>> 1$ will
                yield $y > y_{max}$ at the next timestep.
                Hence, we only aim at guaranteeing the safety of trajectories entering buffer $\mathcal{B}$.
                We use the <a href="#POLICE">POLICE</a> algorithm to make our policy affine inside buffer region
                $\mathcal{B}$.
            </p>
          </div>
          <div class="content has-text-justified">
            <p>
              We implemented our approach on two environments: the Gym inverted pendulum and a Space Shuttle landing
              scenario.
              We trained <a href="#PPO">PPO</a> policies for both tasks with additional negative rewards to promote
              constraint respect.
              We augment the PPO policies with our POLICEd RL to bring constraint violations to zero.
            </p>
          </div>
        </div>
      </div>
    </div>
  </section> -->



  <section class="section hero is-light">
    <div class="container is-max-desktop">
      <div class="columns is-centered has-text-centered">
        <div class="column is-four-fifths">
          <h2 class="title">Overall architecture diagram for our LLM-MCA and LLM-TACA methods</h2>
          <div class="content has-text-justified">
            <p>
              Our centralized training architecture uti-
              lizes a centralized LLM-critic instantiated with our base prompt (en-
              vironment description, our definitions, and task query). At each
              timestep, we update our LLM-critic with the global reward and lat-
              est observations from the environment. We then update our agents’
              policies with the individualized feedback from our critic.
            </p>
          </div>

          <img src="static/images/Overall_Architechture_v13.png" alt="architecture diagram"
            style="height:300px !important; display:block !important; margin:auto !important;" />
        </div>
      </div>
    </div>
  </section>


  <!-- Image carousel -->
  <!-- <section class="hero is-light">
    <div class="hero-body">
      <div class="container">
        <div id="results-carousel" class="carousel results-carousel">
          <div class="item">
            <img src="static/images/shuttle_phase_base_POLICEd.svg" alt="phase portrait"
              style="height:400px !important; display:block !important; margin:auto !important;" />
            <h4 class="subtitle has-text-centered">
              Phase portrait of the Space Shuttle landing with <strong><span style="color:green;">green</span></strong>
              buffer. <br>
              Soft landings correspond to the <strong><span style="color:pink;">pink</span></strong> target region.
            </h4>
          </div>
          <div class="item">
            <img src="static/images/shuttle_h.svg" alt="shuttle altitude"
              style="height:400px !important; display:block !important; margin:auto !important;" />
            <h4 class="subtitle has-text-centered">
              Altitude of the Space Shuttle.
            </h4>
          </div>
          <div class="item">
            <img src="static/images/shuttle_v.svg" alt="shuttle velocity"
              style="height:400px !important; display:block !important; margin:auto !important;" />
            <h4 class="subtitle has-text-centered">
              Velocity of the Space Shuttle.
            </h4>
          </div>
          <div class="item">
            <img src="static/images/shuttle_gamma.svg" alt="shuttle flight path angle"
              style="height:400px !important; display:block !important; margin:auto !important;" />
            <h4 class="subtitle has-text-centered">
              Flight path angle of the Space Shuttle.
            </h4>
          </div>
          <div class="item">
            <img src="static/images/shuttle_alpha.svg" alt="shuttle angle of attack"
              style="height:400px !important; display:block !important; margin:auto !important;" />
            <h4 class="subtitle has-text-centered">
              Angle of attack of the Space Shuttle.
            </h4>
          </div>
        </div>
      </div>
    </div>
  </section>
  <!-- End image carousel -->

  <!-- <section class="hero is-light">
    <div class="container is-max-desktop">
      <div class="hero-body">
        <h4 class="subtitle has-text-centered">
          POLICEd soft landing of the space shuttle from the full 10,000 ft.
        </h4>
        <video poster="" id="tree" autoplay controls muted loop width="556" height="800">
          <source src="static/videos/POLICEd_shuttle_10006.mp4" type="video/mp4">
        </video>
      </div>
    </div>
  </section>



  <section class="section hero is-small">
    <div class="container is-max-desktop">
      <div class="columns is-centered has-text-centered">
        <div class="column is-four-fifths">
          <h2 class="title">Guaranteed stabilization of the inverted pendulum</h2>
          <div class="content has-text-justified">
            <p>
              The difficulty with the inverted pendulum is enlarging the stability region and guaranteeing the
              constraint respect for a wide range of initial conditions.
              Both the baseline PPO and POLICEd version easily stabilize the pendulum when starting near the
              equilibrium, but they differ on more demanding initial conditions.
            </p>
          </div>
          <img src="static/images/pendulum_multi_phase.png" alt="space shuttle"
            style="height:300px !important; display:block !important; margin:auto !important;" />
        </div>
      </div>
    </div>
  </section>

  <section class="hero teaser">
    <div class="container is-max-desktop">
      <div class="hero-body">
        <div style="display: inline">
          <div style="width:300px; display: inline-block; float:left; margin-right: 100px;">
            <h4 class="subtitle has-text-centered">
              <strong>Baseline</strong> PPO controller starting with a high velocity fails.
            </h4>
            <video poster="" id="tree" autoplay controls muted loop width="300" height="300">
              <source src="static/videos/pendulum_baseline.mp4" type="video/mp4">
            </video>
          </div>

          <div style="width:300px; display: inline-block; float:right;">
            <h4 class="subtitle has-text-centered">
              <strong>POLICEd</strong> controller starting from the same state succeeds.
            </h4>
            <video poster="" id="tree" autoplay controls muted loop width="300" height="300">
              <source src="static/videos/pendulum_POLICEd.mp4" type="video/mp4">
            </video>
          </div>
        </div>
      </div>
    </div>
  </section> -->




  <!-- Presentation video-->
  <!--section class="hero is-small is-light">
  <div class="hero-body">
    <div class="container">
      <div class="columns is-centered has-text-centered">
        <div class="column is-four-fifths">
        <h2 class="title is-3">Video Presentation given at RSS 2024</h2>
          <div class="publication-video">
            <iframe width="560" height="315" src="https://www.youtube.com/embed/xMWSqjRrcVc?si=Puj4cTJtLFcLGyH1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"   referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
        </div>
        </div>
      </div>
    </div>
  </div>
</section-->


  <!-- Video carousel -->
  <!-- <section class="hero is-small">
  <div class="hero-body">
    <div class="container">
      <h2 class="title is-3">Another Carousel</h2>
      <div id="results-carousel" class="carousel results-carousel">
        <div class="item item-video1">
          <video poster="" id="video1" autoplay controls muted loop height="100%">
            <!- - Your video file here - ->
            <source src="static/videos/carousel1.mp4"
            type="video/mp4">
          </video>
        </div>
        <div class="item item-video2">
          <video poster="" id="video2" autoplay controls muted loop height="100%">
            <!- - Your video file here - ->
            <source src="static/videos/carousel2.mp4"
            type="video/mp4">
          </video>
        </div>
        <div class="item item-video3">
          <video poster="" id="video3" autoplay controls muted loop height="100%">\
            <!- - Your video file here - ->
            <source src="static/videos/carousel3.mp4"
            type="video/mp4">
          </video>
        </div>
      </div>
    </div>
  </div>
</section> -->
  <!-- End video carousel -->






  <!--BibTex citation -->
  <section class="section" id="BibTeX">
    <div class="container is-max-desktop content">
      <h2 class="title">BibTeX</h2>
      <pre><code>@inproceedings{nagpal2024llmca,
        title={Leveraging Large Language Models for Effective and Explainable Multi-Agent Credit Assignment},
        author={Nagpal, Kartik and Dong, Dayi and Bouvier, Jean-Baptiste and Mehr, Negar},
        booktitle = {24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
        year={2025}
      }</code></pre>
    </div>
  </section>
  <!--End BibTex citation -->

  <!--References -->
  <!-- <section class="section hero is-small is-light">
    <div class="container is-max-desktop content">
      <h2 class="title">References</h2>
      <dl>
        <dt><strong>[Space Shuttle]</strong></dt>
        <dd>
          <div class="reference" id="Shuttle">
            Ali Heydari and S. N. Balakrishnan,
            <a href="https://arc.aiaa.org/doi/pdf/10.2514/6.2011-6641" target="_blank" rel="noopener noreferrer">Optimal
              Online Path Planning for Approach and Landing Guidance</a>,
            AIAA Atmospheric Flight Mechanics Conference, 2011.
          </div>
        </dd>
        <dt><strong>[POLICEd RL]</strong></dt>
        <dd>
          <div class="reference" id="POLICEd RL">
            Jean-Baptiste Bouvier, Kartik Nagpal, and Negar Mehr,
            <a href="https://arxiv.org/pdf/2403.13297.pdf" target="_blank" rel="noopener noreferrer">POLICEd RL:
              Learning Closed-Loop Robot Control Policies with Provable Satisfaction of Hard Constraints</a>,
            Robotics: Science and Systems (RSS), 2024.
          </div>
        </dd>
        <dt><strong>[POLICE]</strong></dt>
        <dd>
          <div class="reference" id="POLICE">
            Randall Balestriero and Yann LeCun,
            <a href="https://ieeexplore.ieee.org/abstract/document/10096520" target="_blank" rel="noopener noreferrer">
              POLICE: Provably optimal linear constraint enforcement for deep neural networks</a>,
            IEEE International Conference on Acoustics, Speech and Signal Processing, 2023.
          </div>
        </dd>
        <dt><strong>[PPO]</strong></dt>
        <dd>
          <div class="reference" id="PPO">
            John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov
            <a href="https://arxiv.org/pdf/1707.06347" target="_blank" rel="noopener noreferrer">Proximal Policy
              Optimization Algorithms</a>,
            OpenAI, 2017.
          </div>
        </dd>

      </dl>
    </div>
  </section> -->



  <footer class="footer">
    <div class="container">
      <div class="columns is-centered">
        <div class="column is-8">
          <div class="content">
            <p>
              This work is supported by the National Science Foundation, under grants ECCS-2145134, CAREER Award,
              CNS-2423130, and CCF-2423131.
            </p>
            <p>
              This page was built using the <a href="https://github.com/eliahuhorwitz/Academic-project-page-template"
                target="_blank">Academic Project Page Template</a> which was adopted from the <a
                href="https://nerfies.github.io" target="_blank">Nerfies</a> project page.
              <br> This website is licensed under a <a rel="license"
                href="http://creativecommons.org/licenses/by-sa/4.0/" target="_blank">Creative
                Commons Attribution-ShareAlike 4.0 International License</a>.
            </p>
          </div>
        </div>
      </div>
    </div>
  </footer>

</body>

</html>