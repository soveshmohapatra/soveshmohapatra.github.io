---
layout: null
permalink: /research/TReND/
---

<html>
<head>
  <meta charset="utf-8">
  <meta name="description" content="TReND: A self-supervised transformer-autoencoder framework integrating regularized NMF for neonatal functional network delineation.">
  <meta property="og:title" content="TReND: Transformer-derived Features and Regularized NMF for Neonatal Functional Network Delineation"/>
  <meta property="og:description" content="A novel self-supervised transformer-autoencoder framework integrating RNMF to unveil neonatal functional networks in rs-fMRI data."/>
  <meta property="og:url" content="https://arxiv.org/abs/YOUR_PAPER_ID"/>
  <meta property="og:image" content="static/image/TReND_paper_banner.png" />
  <meta property="og:image:width" content="1200"/>
  <meta property="og:image:height" content="630"/>
  <meta name="twitter:title" content="TReND: Transformer-derived Features and Regularized NMF for Neonatal Functional Network Delineation">
  <meta name="twitter:description" content="A robust framework for neonatal functional parcellation using transformer-based autoencoder and RNMF clustering."/>
  <meta name="twitter:image" content="static/images/TReND_twitter_banner.png">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="keywords" content="Neonatal Functional Parcellation, Transformer Autoencoder, Resting-state fMRI, Regularized NMF, Functional Connectivity, Deep Learning, Brain Development, Machine Learning, Geodesic Distance, Self-supervised Learning">
  <meta name="viewport" content="width=device-width, initial-scale=1">



  <title>TReND for Neonatal Parcellation</title>
  <link rel="icon" type="image/x-icon" href="/assets/TReND/Icon.png">
  <link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro"
  rel="stylesheet">

  <link rel="stylesheet" href="/assets/TReND/css/bulma.min.css">
  <link rel="stylesheet" href="/assets/TReND/css/bulma-carousel.min.css">
  <link rel="stylesheet" href="/assets/TReND/css/bulma-slider.min.css">
  <link rel="stylesheet" href="/assets/TReND/css/fontawesome.all.min.css">
  <link rel="stylesheet"
  href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  <link rel="stylesheet" href="/assets/TReND/css/index.css">

  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="https://documentcloud.adobe.com/view-sdk/main.js"></script>
  <script defer src="/assets/TReND/js/fontawesome.all.min.js"></script>
  <script src="/assets/TReND/js/bulma-carousel.min.js"></script>
  <script src="/assets/TReND/js/bulma-slider.min.js"></script>
  <script src="/assets/TReND/js/index.js"></script>
</head>
<body>


  <section class="hero">
    <div class="hero-body">
      <div class="container is-max-desktop">
        <div class="columns is-centered">
          <div class="column has-text-centered">
            <h1 class="title is-1 publication-title">TReND: Transformer-derived features and Regularized NMF for neonatal functional network Delineation</h1>
            <div class="is-size-5 publication-authors">
              <!-- Paper authors -->
              <span class="author-block">
                <a href="www.soveshmohapatra.com" target="_blank">Sovesh Mohapatra</a>,</span>
                <span class="author-block">
                  <a href="SECOND AUTHOR PERSONAL LINK" target="_blank">Second Author</a><sup>*</sup>,</span>
                  <span class="author-block">
                    <a href="THIRD AUTHOR PERSONAL LINK" target="_blank">Third Author</a>
                  </span>
                  </div>

                  <div class="is-size-5 publication-authors">
                    <span class="author-block">Institution Name<br>Conferance name and year</span>
                    <span class="eql-cntrb"><small><br><sup>*</sup>Indicates Equal Contribution</small></span>
                  </div>

                  <div class="column has-text-centered">
                    <div class="publication-links">
                         <!-- Arxiv PDF link -->
                      <span class="link-block">
                        <a href="https://arxiv.org/pdf/<ARXIV PAPER ID>.pdf" target="_blank"
                        class="external-link button is-normal is-rounded is-dark">
                        <span class="icon">
                          <i class="fas fa-file-pdf"></i>
                        </span>
                        <span>Paper</span>
                      </a>
                    </span>

                    <!-- Supplementary PDF link -->
                    <span class="link-block">
                      <a href="static/pdfs/supplementary_material.pdf" target="_blank"
                      class="external-link button is-normal is-rounded is-dark">
                      <span class="icon">
                        <i class="fas fa-file-pdf"></i>
                      </span>
                      <span>Supplementary</span>
                    </a>
                  </span>

                  <!-- Github link -->
                  <span class="link-block">
                    <a href="https://github.com/YOUR REPO HERE" target="_blank"
                    class="external-link button is-normal is-rounded is-dark">
                    <span class="icon">
                      <i class="fab fa-github"></i>
                    </span>
                    <span>Code</span>
                  </a>
                </span>

                <!-- ArXiv abstract Link -->
                <span class="link-block">
                  <a href="https://arxiv.org/abs/<ARXIV PAPER ID>" target="_blank"
                  class="external-link button is-normal is-rounded is-dark">
                  <span class="icon">
                    <i class="ai ai-arxiv"></i>
                  </span>
                  <span>arXiv</span>
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
            Precise parcellation of functional networks (FNs) of early developing human brain is the fundamental basis for identifying biomarker of developmental disorders and understanding functional development. Resting-state fMRI (rs-fMRI) enables in vivo exploration of functional changes, but adult FN parcellations cannot be directly applied to the neonates due to incomplete network maturation. No standardized neonatal functional atlas is currently available.  To solve this fundamental issue, we propose TReND, a novel and fully automated self-supervised transformer-autoencoder framework that integrates regularized nonnegative matrix factorization (RNMF) to unveil the FNs in neonates. TReND effectively disentangles spatiotemporal features in voxel-wise rs-fMRI data. The framework integrates confidence-adaptive masks into transformer self-attention layers to mitigate noise influence. A self supervised decoder acts as a regulator to refine the encoder's latent embeddings, which serve as reliable temporal features. For spatial coherence, we incorporate brain surface-based geodesic distances as spatial encodings along with functional connectivity from temporal features. The TReND clustering approach processes these features under sparsity and smoothness constraints, producing robust and biologically plausible parcellations. We extensively validated our TReND framework on three different rs-fMRI datasets: simulated, dHCP and HCP-YA against comparable traditional feature extraction and clustering techniques. Our results demonstrated the superiority of the TReND framework in the delineation of neonate FNs with significantly better spatial contiguity and functional homogeneity. Collectively, we established TReND, a novel and robust framework, for neonatal FN delineation. TReND-derived neonatal FNs could serve as a neonatal functional atlas for perinatal populations in health and disease.
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
        <!-- Your image here -->
        <img src="static/images/carousel1.jpg" alt="MY ALT TEXT"/>
        <h2 class="subtitle has-text-centered">
          First image description.
        </h2>
      </div>
      <div class="item">
        <!-- Your image here -->
        <img src="static/images/carousel2.jpg" alt="MY ALT TEXT"/>
        <h2 class="subtitle has-text-centered">
          Second image description.
        </h2>
      </div>
      <div class="item">
        <!-- Your image here -->
        <img src="static/images/carousel3.jpg" alt="MY ALT TEXT"/>
        <h2 class="subtitle has-text-centered">
         Third image description.
       </h2>
     </div>
     <div class="item">
      <!-- Your image here -->
      <img src="static/images/carousel4.jpg" alt="MY ALT TEXT"/>
      <h2 class="subtitle has-text-centered">
        Fourth image description.
      </h2>
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
      <pre><code>BibTex Code Here</code></pre>
    </div>
</section>
<!--End BibTex citation -->


  <footer class="footer">
  <div class="container">
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content">

          <p>
            This page was built using the <a href="https://github.com/eliahuhorwitz/Academic-project-page-template" target="_blank">Academic Project Page Template</a> which was adopted from the <a href="https://nerfies.github.io" target="_blank">Nerfies</a> project page.
          </p>

        </div>
      </div>
    </div>
  </div>
</footer>

<!-- Statcounter tracking code -->
  
<!-- You can add a tracker to track page visits by creating an account at statcounter.com -->

    <!-- End of Statcounter Code -->

  </body>
  </html>