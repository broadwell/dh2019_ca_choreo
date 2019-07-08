---
title: "Automated Movement and Choreography Analysis of Video Data via Deep Learning Pose Detection"
author: "Peter Broadwell"
date: "7/9/2019"
output: 
  html_document: 
    keep_md: yes
    toc: yes
---



## Initial setup

If you'd like to run this code during the workshop, you'll need to have R and RStudio installed on your computer. If you haven't already done this, please follow the instructions at the bottom of [this page](http://reproducible-science-curriculum.github.io/2015-09-24-reproducible-science-duml/) (credit to the Software Carpentries).

**Important:** here is [the original R Markdown version of this file](https://raw.githubusercontent.com/broadwell/dh2019_ca_choreo/master/DH2019_Movement_Choreography.Rmd), which you can download and open in RStudio.

## Want to generate your own pose source data?

We will be using pre-generated CSV files containing per-frame pose detection output. It is quite difficult to install and run a version of either OpenPose or DensePose that allows you to produce pose detection output in a computationally friendly format, such as JSON. If you're curious, though, these GitHub repositories contain utilities to do so:

- OpenPose: <https://github.com/UCLAXLabs/tf-pose-estimation/tree/ucla_kpop>

- DensePose: <https://github.com/UCLAXLabs/DensePose/tree/ucla_kpop>

Also, this script can convert the DensePose JSON output to the CSV format used in this workshop:

- <https://github.com/tango2/kpopspace/blob/master/video/figuresJSONtoCSV.py>

And here's the link to the Google Colaboratory notebook that walks through the rather arduous steps of downloading, compiling and running the OpenPose demo on a sample video:  
<http://bit.ly/dh2019-ca-openpose>  
After opening the notebook in Colab, you'll need to click `File->Save a Copy in Drive...` to get a local copy that you can run.

## Install required libraries

If you don't have all of these packages installed, you'll need to run some or all of the first line (without the #) on your local Rstudio console:


```r
#install.packages(c("distances","ade4","ggplot2","naniar","reshape2","deldir","mdpeer","BBmisc","lubridate"))

library(ggplot2)

# For distance matrix-based comparison
library(distances)
library(ade4)
library(naniar)
library(reshape2)

# For adjacency matrix-based comparison (using Delaunay triangulation)
library(deldir)
library(mdpeer)
library(BBmisc)
library(lubridate)
```

## Download pose data files

A cover of Jennie's "SOLO" by Lisa Rhee (25 fps):  
<https://www.youtube.com/watch?v=Zu3hBEZ0RvA>


```r
download.file("https://docs.google.com/uc?export=download&id=1gZhXTlQU-JTplu4mBRJLhWUxGNc6VZKQ", "solo_figures.csv")
```

The final 41 seconds of Taemin's "Drip Drop" performance video (solo, 24 fps):  
<https://www.youtube.com/watch?v=Oz3mm3tPKfg&t=3m9s>


```r
download.file("https://docs.google.com/uc?export=download&id=1mzk6hPEkq8OeUrFuTgUNO2VbmEagxBs2", "Taemin_Drip_Drop_solo.csv")
```

The dance practice video for BTS's "Fire" (multi-dancer, 30 fps, downsampled from 60 fps):  
<https://www.youtube.com/watch?v=sWuYspuN6U8>  
NOTE: The data for the entire video is quite large (~20 MB) and takes a long time to process, so we'll use an excerpt for now.


```r
# This gets the entire file
#download.file("https://docs.google.com/uc?export=download&id=16k6oCGzL7gfFKhWIzVi-MAGb5BMjsOif", "fire_figures.csv")
download.file("https://docs.google.com/uc?export=download&id=1dEUxcvnOvZgX6mu8ljzW8C20ov4rWNd3", "fire_excerpt.csv")
```

## Define helper functions

These functions include procedures for creating a single-pose distance matrix and its Laplacian matrix equivalent, as well as methods that return a value that quantifies the similarity/difference between two such pose representations.


```r
# Define empty values so that we can ignore rows that contain them
na_strings <- c("NA", "N A", "N / A", "N/A", "N/ A")

# Assemble a data frame of the position data for a single detected figure at a point in time
frametocoords <- function(FiguresRow) {
  df <- data.frame(x = c(FiguresRow$nose_x, FiguresRow$left_eye_x, FiguresRow$right_eye_x, FiguresRow$left_ear_x, FiguresRow$right_ear_x, FiguresRow$left_shoulder_x, FiguresRow$right_shoulder_x, FiguresRow$left_elbow_x, FiguresRow$right_elbow_x, FiguresRow$left_wrist_x, FiguresRow$right_wrist_x, FiguresRow$left_hip_x, FiguresRow$right_hip_x, FiguresRow$left_knee_x, FiguresRow$right_knee_x, FiguresRow$left_ankle_x, FiguresRow$right_ankle_x),
                   y = c(FiguresRow$nose_y, FiguresRow$left_eye_y, FiguresRow$right_eye_y, FiguresRow$left_ear_y, FiguresRow$right_ear_y, FiguresRow$left_shoulder_y, FiguresRow$right_shoulder_y, FiguresRow$left_elbow_y, FiguresRow$right_elbow_y, FiguresRow$left_wrist_y, FiguresRow$right_wrist_y, FiguresRow$left_hip_y, FiguresRow$right_hip_y, FiguresRow$left_knee_y, FiguresRow$right_knee_y, FiguresRow$left_ankle_y, FiguresRow$right_ankle_y),
                   grp = c(1, 1, 1, 1, 1, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5),
                   figure_index = FiguresRow$figure_index,
                   frame_id = FiguresRow$frame_id,
                   seconds = FiguresRow$frame_timecode)
  return(df)
}

# Convert figure location data to its Laplacian matrix representation
getlap <- function(df, normalize = TRUE) {
  dxy <- deldir(df$y, df$x)
  ind <- dxy$delsgs[,5:6] 
  adj <- matrix(0, length(df$x), length(df$y)) 
  for (i in 1:nrow(ind)){ 
    adj[ind[i,1], ind[i,2]] <- 1 
    adj[ind[i,2], ind[i,1]] <- 1 
  }
  lap <- Adj2Lap(adj)
  if (normalize == TRUE) {
    lap <- L2L.normalized(lap)
  }
  return(lap)
}

# Compare the body part distance matrices of two figures
comparerows <- function(row1, row2){
  row1coords <- frametocoords(row1)
  row1dist <- distances(row1coords, dist_variables=c("x","y"), normalize="mahalanobize")
  row2coords <- frametocoords(row2)
  row2dist <- distances(row2coords, dist_variables=c("x","y"), normalize="mahalanobize")
  res <- mantel.rtest(as.dist(row1dist), as.dist(row2dist))
  return(res[["obs"]])
}

# Compare the Laplacian (graph-based) matrices for two figures
comparelaps <- function(row1, row2, normalize = TRUE){
  row1coords <- frametocoords(row1)
  row1lap <- getlap(row1coords, normalize)
  row2coords <- frametocoords(row2)
  row2lap <- getlap(row2coords, normalize)
  diff <- sum(abs(row1lap - row2lap))
  return(diff)
}

# Determine the effective frames per second of the pose detection output
getfps <- function(Figs) {
  first_timecode <- Figs[1,]$frame_timecode
  last_timecode <- Figs[nrow(Figs),]$frame_timecode
  unique_frames <- unique(Figs$frame_id)
  total_frames <- length(unique_frames)
  duration_in_seconds <- last_timecode - first_timecode
  #frame_extent <- last_frame - first_frame
  #full_fps <- frame_extent / duration_in_seconds
  fps <- total_frames / duration_in_seconds
  return(fps)
}
```

## Task 1: Visualize and explore choreography self-similarity (singe-dancer time series)


```r
# Procedure for single-dancer analysis of choreography self-similarity (across time)
# method is either "distance" or "laplacian"
# If step_frames is set to the video frame rate, this produces ~one observation per second
selfcomparison <- function(Figs, method="distance", step_frames=1) {
  
  unique_frames <- unique(Figs$frame_id)
  total_frames <- length(unique_frames)
  observations = as.integer(total_frames / step_frames)
  
  fps <- getfps(Figs)

  sm <- matrix(nrow=observations,ncol=observations,byrow=TRUE)

  frame_i <- 1
  for(i in seq(1,observations)) {
    #print(paste("comparing",i,"of",observations))
    frame_j <- 1
    for(j in seq(1,observations)) {
      if (i == j) {
        sm[i,j] <- 1
      } else if (i < j) {
        rowA <- Figs[frame_i,]
        rowB <- Figs[frame_j,]
        if (method == "laplacian") {
          obs <- comparelaps(rowA, rowB)
        } else { #if (method == "laplacian")
          obs <- comparerows(rowA, rowB)
        }
        sm[i,j] <- obs
        sm[j,i] <- obs
      }
      frame_j <- j * step_frames
    }
    frame_i <- i * step_frames
  }
  if (method == "laplacian") {
    # Normalize and invert the Laplacian comparisons so they'll look like the other results
    sm <- apply(sm, MARGIN = 2, FUN = function(X) (1 - (X - min(X))/diff(range(X))))
  }
  
  ggplot(melt(sm), aes(Var1, Var2, fill=value)) +
    geom_tile(height=1, width=1) +
    scale_fill_gradient2(low="blue", mid="white", high="red") +
    theme_minimal() +
    coord_equal() +
    labs(x="",y="",fill="Corr") +
    scale_x_time(labels = function(l) { seconds_to_period(round(period_to_seconds(hms(l))/(fps/step_frames)))}) +
    scale_y_time(labels = function(l) { seconds_to_period(round(period_to_seconds(hms(l))/(fps/step_frames)))}) +
    theme(axis.text.x=element_text(size=13, angle=45, vjust=1, hjust=1, 
                                   margin=margin(-3,0,0,0)),
          axis.text.y=element_text(size=13, margin=margin(0,-3,0,0)))
}
```

## Task 2: Quantify synchronized movement among multiple dancers


```r
# Procedure for computing per-frame figure similarity for multi-dancer videos
# method is either "distance" or "laplacian"
# step_frames can reduce computation -- this many frames are skipped between comparisons
# max_figures is necessary because including backup dancers takes too long
multifigcompare <- function(Figs, method="distance", step_frames=5, max_figures=7) {
  mean_frame_sims <- numeric()
  sim_stdevs <- numeric()
  time_labels <- numeric()
  unique_frames <- unique(Figs$frame_id)
  for(i in sort(unique_frames)) {
    frame_figs <- Figs[Figs$frame_id == i,]
    
    ff_similarities = numeric()
    mean_similarity <- 0
    stdev <- 0
    time_code <- frame_figs[1,]$frame_timecode
    if ((i %% step_frames == 0) && (nrow(frame_figs) > 1) && (nrow(frame_figs) <= max_figures)) {
      
      # Compare every figure to every other figure 
      for(j in frame_figs$figure_index) {
        for(k in frame_figs$figure_index) {
          if (j < k) {
            rowA <- frame_figs[frame_figs$figure_index == j,]
            rowB <- frame_figs[frame_figs$figure_index == k,]
            if (method == "laplacian") {
              obs <- comparelaps(rowA, rowB)
            } else { # (method == "distance")
              obs <- comparerows(rowA, rowB)
            }
            ff_similarities <- c(ff_similarities, obs)
          }
        }
      }
      #print(paste("frame ",i,"has",nrow(frame_figs),"figures"))
      mean_similarity <- mean(ff_similarities)
      stdev = sd(ff_similarities)
    }
    mean_frame_sims <- c(mean_frame_sims, mean_similarity)
    sim_stdevs <- c(sim_stdevs, stdev)
    time_labels <- c(time_labels, time_code)
  }
  
  # Note that the Laplacian similarity values are inverted (closer to 0 = more similar),
  # so the graph output also will be inverted, compared to the distance method
  graph_data <- data.frame(sims=mean_frame_sims, sd=sim_stdevs, secs=as.difftime(time_labels, units = 'secs'))
  ggplot(graph_data, aes(x=secs, y=mean_frame_sims)) + geom_errorbar(aes(ymin=mean_frame_sims-sd, ymax=mean_frame_sims+sd), width=.1, color='black') + 
    geom_point(color='blue') + scale_x_time(labels = function(l) strftime(l, '%M:%S'))
}
```

## Try it out!

The following code blocks run the analytical tasks defined above on excerpts of videos and generate visualizations of their results. Feel free to experiment with the settings, or with the code itself.

First, a choreography self-comparison analysis on a solo video. The output is a correlation matrix heatmap visualization, which compares the similarity of every analyzed frame of the video to every other frame.

```r
Figures <- read.csv("solo_figures.csv", na = na_strings)
Figs <- Figures[complete.cases(as.data.frame(Figures)),]
selfcomparison(Figs, "laplacian", step_frames=24)
```

![](DH2019_Movement_Choreography_files/figure-html/solo_comparison-1.png)<!-- -->

Another solo self-comparison analysis, but with a considerably faster-paced performance video. Increasing the frame sample rate therefore is prudent, and it's a shorter video, so we can use the longer-running but more accurate distance-based similarity metric.

```r
Figures <- read.csv("Taemin_Drip_Drop_solo.csv", na = na_strings)
Figs <- Figures[complete.cases(as.data.frame(Figures)),]
selfcomparison(Figs, "distance", step_frames=6)
```

![](DH2019_Movement_Choreography_files/figure-html/another_solo-1.png)<!-- -->

Finally, a per-frame pose similarity comparison across all dancers in a multi-dancer video. The output is a time-series plot of the average pose similarity values for each analyzed frame, with bars indicating the standard deviation around each mean.

```r
#Figures <- read.csv("fire_figures.csv", na = na_strings)
Figures <- read.csv("fire_excerpt.csv", na = na_strings)
Figs <- Figures[complete.cases(as.data.frame(Figures)),]
multifigcompare(Figs, "distance", step_frames=6)
```

![](DH2019_Movement_Choreography_files/figure-html/multifig_comparison-1.png)<!-- -->
