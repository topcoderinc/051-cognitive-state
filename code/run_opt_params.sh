docker build -t 'latest' .

docker run -v /Users/tearth/Documents/GitHub/ltfs_hack/incident-insights/topcoder_cognitive_state/data/:/work/data/ \
           -v /Users/tearth/Documents/GitHub/ltfs_hack/incident-insights/topcoder_cognitive_state/sample-submission/code/model/:/work/model/ \
           latest sh opt_params.sh /work/data/data_training.zip /work/data/log.txt
