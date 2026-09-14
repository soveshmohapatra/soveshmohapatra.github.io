---
layout: null
permalink: /research/connectome-compiler/
---

<html>
<head>
  <meta charset="utf-8">
  <meta name="description" content="Connectome Response Analysis and Design (CoRAD): an analytic framework that reads order-specific responses from a connectome, decompiles them into linear-memory and nonlinear-mixing dimensions, and uses their gradients to write constrained changes back into the wiring.">
  <meta property="og:title" content="Connectome as Compiler: Reading Computation from Connectivity and then Writing it Back"/>
  <meta property="og:description" content="CoRAD links recurrent connectivity to predicted temporal-processing capacities and guides constrained rewiring toward specified memory and mixing targets."/>
  <meta property="og:url" content="https://soveshmohapatra.com/research/connectome-compiler/"/>
  <meta property="og:image" content="/assets/connectome-compiler/Icon.png" />
  <meta property="og:image:width" content="1200"/>
  <meta property="og:image:height" content="630"/>
  <meta name="twitter:title" content="Connectome as Compiler: Reading Computation from Connectivity and then Writing it Back">
  <meta name="twitter:description" content="CoRAD reads memory and mixing capacities from a connectome and writes constrained changes back into the wiring."/>
  <meta name="twitter:image" content="/assets/connectome-compiler/Icon.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="keywords" content="Connectome, Reservoir Computing, Network Neuroscience, Linear Memory, Nonlinear Mixing, Controllability Gramian, Structural Connectivity, Human Connectome Project, Network Design, Recurrent Dynamics, CoRAD">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <title>Connectome as Compiler</title>
  <link rel="icon" href="data:,">
  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"
  rel="stylesheet">

  <link rel="stylesheet" href="/assets/connectome-compiler/css/bulma.min.css">
  <link rel="stylesheet" href="/assets/connectome-compiler/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="/assets/connectome-compiler/css/bulma-slider.min.css">
  <link rel="stylesheet" href="/assets/connectome-compiler/css/fontawesome.all.min.css">
  <link rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="/assets/connectome-compiler/css/index.css">

  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script defer src="/assets/connectome-compiler/js/fontawesome.all.min.js"></script>
  <script src="/assets/connectome-compiler/js/bulma-carousel.min.js"></script>
  <script src="/assets/connectome-compiler/js/bulma-slider.min.js"></script>
  <script src="/assets/connectome-compiler/js/index.js"></script>
</head>
<body>


  <section class="hero">
    <div class="hero-body">
      <div class="container is-max-desktop">
        <div class="columns is-centered">
          <div class="column has-text-centered">
            <h2 class="title is-1 publication-title">Connectome as Compiler: Reading Computation from Connectivity and then Writing it Back</h2>
            <div class="is-size-5 publication-authors">
              <!-- Paper authors -->
              <span class="author-block">
                Sovesh Mohapatra, Dani S. Bassett</span>
            </div>

            <div class="is-size-5 publication-authors">
              <span class="author-block">Host Institution: University of Pennsylvania</span>
            </div>

            <div class="column has-text-centered">
              <div class="publication-links">

                <!-- Paper (PDF) -->
                <span class="link-block">
                  <a href="/assets/connectome-compiler/Paper.pdf" target="_blank"
                  class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                    <i class="fas fa-file-pdf"></i>
                  </span>
                  <span>Paper</span>
                </a>
              </span>

              <!-- Supplementary Information (PDF) -->
              <span class="link-block">
                <a href="/assets/connectome-compiler/SI.pdf" target="_blank"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon">
                  <i class="fas fa-file-alt"></i>
                </span>
                <span>Supplementary Information</span>
              </a>
            </span>

              <!-- arXiv (coming soon) -->
              <span class="link-block">
                <a class="external-link button is-normal is-rounded is-dark" disabled style="pointer-events: none; opacity: 0.65;">
                <span class="icon">
                  <i class="ai ai-arxiv"></i>
                </span>
                <span>arXiv (Coming soon)</span>
              </a>
            </span>

              <!-- GitHub (coming soon) -->
              <span class="link-block">
                <a class="external-link button is-normal is-rounded is-dark" disabled style="pointer-events: none; opacity: 0.65;">
                <span class="icon">
                  <i class="fab fa-github"></i>
                </span>
                <span>Code (Coming soon)</span>
              </a>
            </span>

        </div>
      </div>
    </div>
  </div>
</div>
</div>
</section>


<!-- Paper abstract -->
<section class="section hero is-light">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified">
          <p>
            Neural systems use information from earlier inputs to shape their current responses. How recurrent connectivity supports the retention and combination of this information remains difficult to quantify. Here, we introduce <b>Co</b>nnectome <b>R</b>esponse <b>A</b>nalysis and <b>D</b>esign (CoRAD), an analytic framework that derives order-specific responses from a connectome and decompiles them into dimensions computationally associated with linear memory and nonlinear mixing. Each dimension is an effective count of the response directions generated at its corresponding order, providing a compact summary of how past inputs shape network activity. CoRAD uses their gradients to guide constrained changes to existing connections. Across a range of networks&mdash;from synthetic reservoirs, to human and non-human connectomes&mdash;the response dimensions predicted their corresponding simulated memory capacities in held-out data. Their decompositions showed how response contributions were distributed across input histories and cortical regions. Gradient-guided reweighting changed the corresponding held-out capacities and moved one or both dimensions toward specified targets under wiring constraints. Most of the gain found by the constrained searches was recovered using only a small fraction of existing connections, while different reweightings reached similar joint targets. Together, CoRAD provides a quantitative link between recurrent connectivity and the explanation, interpretation, and constrained design of predicted temporal-processing capacities.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>
<!-- End paper abstract -->


<!-- Image carousel -->
<section class="hero is-small">
  <div class="hero-body">
    <div class="container">
      <div id="results-carousel" class="carousel results-carousel">
        <div class="item">
          <img src="/assets/connectome-compiler/Figure1.png" alt="CoRAD overview: read, decompile, write" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            CoRAD reads input-evoked responses from a connectome, decompiles them into memory and mixing dimensions, and writes constrained changes back into the wiring.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/connectome-compiler/Figure2.png" alt="Response dimensions predict simulated capacities" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            Response dimensions predict simulated memory and mixing capacities in held-out synthetic reservoirs, human connectomes, and non-human connectomes.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/connectome-compiler/Figure3.png" alt="Mode- and target-specific decompositions" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            Response dimensions decompose into recoverable memory and mixing profiles across input delays and delayed products.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/connectome-compiler/Figure4.png" alt="Regional response profiles" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            Regional response profiles correspond to the memory and mixing capacities accessible from cortical readouts and operating states.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/connectome-compiler/Figure5.png" alt="Response-gradient reweighting" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            Response-gradient reweighting changes simulated memory and mixing capacities in human connectomes.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/connectome-compiler/Figure6.png" alt="Constrained reweighting toward targets" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            Constrained reweighting reaches specified response targets, with most of the gain carried by a small fraction of existing connections.
          </h4>
        </div>
      </div>
    </div>
  </div>
</section>
<!-- End image carousel -->


<!--BibTex citation -->
  <section class="section" id="BibTeX">
    <div class="container is-max-desktop content">
      <h2 class="title">BibTeX</h2>
      <pre><code>@article{mohapatra2026connectomecompiler,
  title   = {Connectome as Compiler: Reading Computation from Connectivity and then Writing it Back},
  author  = {Sovesh Mohapatra and Dani S. Bassett},
  year    = {2026},
  note    = {Preprint},
}</code></pre>
    </div>
</section>
<!--End BibTex citation -->


  <footer class="footer">
  <div class="container">
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content">

          <p>
            This page was built using the <a href="https://github.com/eliahuhorwitz/Academic-project-page-template" target="_blank">Academic Project Page Template</a> which was adopted from the <a href="https://nerfies.github.io" target="_blank">Nerfies</a> project page.
          </p>

        </div>
      </div>
    </div>
  </div>
</footer>

  </body>
  </html>
