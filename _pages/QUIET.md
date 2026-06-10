---
layout: null
permalink: /research/QUIET/
---

<html>
<head>
  <meta charset="utf-8">
  <meta name="description" content="QUIET: an edge-centric network control framework that integrates structural controllability and mutual information to identify energy-efficient synchronization pathways in brain networks.">
  <meta property="og:title" content="QUIET: Quantifying Underutilized Influential Edges for Targeted Synchronization"/>
  <meta property="og:description" content="An edge-centric framework integrating structural control theory and information-theoretic functional connectivity to find energy-efficient synchronization pathways."/>
  <meta property="og:url" content="https://soveshmohapatra.com/research/QUIET/"/>
  <meta property="og:image" content="/assets/QUIET/Icon.png" />
  <meta property="og:image:width" content="1200"/>
  <meta property="og:image:height" content="630"/>
  <meta name="twitter:title" content="QUIET: Quantifying Underutilized Influential Edges for Targeted Synchronization">
  <meta name="twitter:description" content="Edge-centric network control for energy-efficient synchronization of brain networks."/>
  <meta name="twitter:image" content="/assets/QUIET/Icon.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="keywords" content="Network Control Theory, Edge Controllability, Mutual Information, Kuramoto, Synchronization, Brain Networks, Line Graph, Connectome, Human Connectome Project, Functional Connectivity">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <title>QUIET: Targeted Synchronization</title>
  <link rel="icon" type="image/x-icon" href="/assets/QUIET/Icon.png">
  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"
  rel="stylesheet">

  <link rel="stylesheet" href="/assets/QUIET/css/bulma.min.css">
  <link rel="stylesheet" href="/assets/QUIET/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="/assets/QUIET/css/bulma-slider.min.css">
  <link rel="stylesheet" href="/assets/QUIET/css/fontawesome.all.min.css">
  <link rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="/assets/QUIET/css/index.css">

  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script defer src="/assets/QUIET/js/fontawesome.all.min.js"></script>
  <script src="/assets/QUIET/js/bulma-carousel.min.js"></script>
  <script src="/assets/QUIET/js/bulma-slider.min.js"></script>
  <script src="/assets/QUIET/js/index.js"></script>
</head>
<body>


  <section class="hero">
    <div class="hero-body">
      <div class="container is-max-desktop">
        <div class="columns is-centered">
          <div class="column has-text-centered">
            <h2 class="title is-1 publication-title">QUIET: Quantifying Underutilized Influential Edges for Targeted Synchronization</h2>
            <div class="is-size-5 publication-authors">
              <!-- Paper authors -->
              <span class="author-block">
                Sovesh Mohapatra, Christoffer G. Alexandersen, Panos Fotiadis, Max Kelz, John Detre, Fabio Pasqualetti, Dani S. Bassett</span>
            </div>

            <div class="is-size-5 publication-authors">
              <span class="author-block">Host Institution: University of Pennsylvania</span>
            </div>

            <div class="column has-text-centered">
              <div class="publication-links">

                <!-- Download Software (gated by request form) -->
                <span class="link-block">
                  <a href="/assets/QUIET/RequestSoftware.html" target="_blank"
                  class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                    <i class="fas fa-download"></i>
                  </span>
                  <span>Download Software</span>
                </a>
              </span>

              <!-- Live Demo (Hugging Face Space) -->
              <span class="link-block">
                <a href="https://huggingface.co/spaces/Sovesh/quiet-live-demo" target="_blank"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon">
                  <i class="fas fa-flask"></i>
                </span>
                <span>Live Demo</span>
              </a>
            </span>

              <!-- NetSci'26 Poster -->
              <span class="link-block">
                <a href="/assets/QUIET/NetSci-QUIET.pdf" target="_blank"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon">
                  <i class="fas fa-image"></i>
                </span>
                <span>NetSci'26 Poster</span>
              </a>
            </span>

              <!-- Paper link (arXiv) -->
              <span class="link-block">
                <a href="https://arxiv.org/pdf/2606.11091" target="_blank"
                class="external-link button is-normal is-rounded is-dark">
                <span class="icon">
                  <i class="fas fa-file-pdf"></i>
                </span>
                <span>Paper</span>
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
            Network control theory can be used to model intrinsic and extrinsic strategies to steer neural dynamics. Standard approaches are node-centric, structural, and focused on achieving desired instantaneous states. Here, we develop an edge-centric approach which incorporates both structure and function to achieve extended patterns of neural dynamics characterized by desired synchronization states. Our method, Quantifying Underutilized Influential Edges for Targeted Synchronization (QUIET), is an edge-centric framework that integrates structural controllability of individual white matter connections and mutual information between pairwise functional timeseries to identify energy-efficient synchronization pathways. QUIET identifies <i>quiet highways</i>, edges that are structurally influential but functionally underutilized, to optimize regional synchronization. We validated QUIET across 75 synthetic configurations, where QUIET-ranked edge sets significantly outperformed random selection in 93% of cases (p &lt; 0.01). The framework, tested on the Human Connectome Project participants, revealed that the control energy required for synchronization of the salience network correlates with fluid intelligence. QUIET, applied to healthy adults undergoing dexmedetomidine-induced unresponsiveness, showed that the frontoparietal and default-mode network exhibited the largest control energy required for synchronization in both awake and sedated states. QUIET is released as a stand-alone software to be used to study theoretically-defined synchronization pathways, which in turn could inform testable hypotheses in perturbative studies.
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
          <img src="/assets/QUIET/Figure1.png" alt="QUIET edge-level control" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            QUIET reframes network control as an edge-level problem.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/QUIET/Figure2.png" alt="Three-stage optimization pipeline" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            QUIET integrates structural controllability and mutual information through a three-stage optimization pipeline.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/QUIET/Figure3.png" alt="Generalization across topologies" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            QUIET generalizes across network topologies, scales, and coupling regimes.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/QUIET/Figure4.png" alt="HCP cognition and emotion" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            QUIET reveals network-specific energy signatures of cognition and emotion in the Human Connectome Project.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/QUIET/Figure5.png" alt="Anesthesia control energy" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            QUIET reveals that anesthetic-induced unconsciousness increases the control energy required for cortical synchronization.
          </h4>
        </div>
        <div class="item">
          <img src="/assets/QUIET/Figure6.png" alt="QUIET pipeline overview" style="display: block; margin: 0 auto; width: 80%;"/>
          <h4 class="subtitle has-text-centered">
            Overview of the QUIET pipeline and its core computational stages.
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
      <pre><code>@article{mohapatra2026quietquantifyingunderutilizedinfluential,
  title         = {QUIET: Quantifying Underutilized Influential Edges for Targeted Synchronization},
  author        = {Sovesh Mohapatra and Christoffer G. Alexandersen and Panagiotis Fotiadis and Max B. Kelz and John A. Detre and Fabio Pasqualetti and Dani S. Bassett},
  year          = {2026},
  eprint        = {2606.11091},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  url           = {https://arxiv.org/abs/2606.11091},
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
