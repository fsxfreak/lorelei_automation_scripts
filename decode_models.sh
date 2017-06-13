#!/usr/bin/env bash

#This script was written by barret zoph for questions email barretzoph@gmail.com

#Set Script Name variable
SCRIPT=`basename ${BASH_SOURCE[0]}`
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )""/"

#Initialize variables to default values.
INPUT_FILE="" #Hard path and name of the source file being rescored (e.g. "/home/nlg/source_data.txt")
TRAINED_MODELS_PATH=""
KBEST_SIZE=""
OUTPUT_FILE=""
LOG_FILE=$PWD/logfile
BEAM_SIZE="" #make sure beam size is >= kbest_size
LONGEST_SENT=""
MODEL_NUMS="1_5_2_6_3_7_4_8" # out of order to avoid putting all the big models on the same gpu
EXTRA_RNN_ARGS="" # user-passed arguments to the RNN binary
QSUBOPTS="" # extra options to pass to qsubrun
RNN_LOCATION="${DIR}/helper_programs/ZOPH_RNN"
NOQSUB=0 # if set to 1, run locally

#Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

#Help function
function HELP {
    echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
    echo "The format that the output file will be in is the following:" 
    echo "${BOLD}The following switches are:${NORM}"
    echo "${REV}--input_file${NORM}  : Specify the location of the data you want to decode."
    echo "${REV}--trained_models${NORM}  : Specify the location of the model you want to decode."
    echo "${REV}--model_nums${NORM}  : Specify the subset of models (1-8, joined with underscore) you want to use (default is all 8)."
    echo "${REV}--num_best${NORM}  : Per sentence in the input_file, this is the number of decodings the model will output." 
    echo "${REV}--output_file${NORM}  : Specify the location of the output file the model generates." 
    echo "${REV}--log_file${NORM}  : Specify the location of the log file the rnn generates." 
    echo "${REV}--extra_rnn_args${NORM}  : Specify any additional argument string to pass to the RNN binary."
    echo "${REV}--rnn_location${NORM}  : Location of the RNN binary."
    echo "${REV}--qsubopts${NORM}  : Specify any additional option string to pass to qsubrun."
    echo "${REV}--noqsub${NORM}  : Run locally. qsubopts are ignored."
    echo -e "${REV}-h${NORM}  : Displays this help message. No further functions are performed."\\n
    echo "Example: ${BOLD}$SCRIPT path/to/decode_models.sh --source_file <source file> --trained_model <path to models> --k_size <number of decodings> --output_file <name and location of output file>  ${NORM}" 
    exit 1
}

#Check the number of arguments. If none are passed, print help and exit.
NUMARGS=$#

echo "Number of arguements specified: ${NUMARGS}"

if [[ $NUMARGS -lt 8 ]]; then
    echo -e \\n"Number of arguments: $NUMARGS"
    HELP
fi

optspec=":h-:"
VAR="1"
NUM=$(( 1 > $(($NUMARGS/2)) ? 1 : $(($NUMARGS/2)) ))
while [[ $VAR -le $NUM ]]; do
    while getopts $optspec FLAG; do 
	case $FLAG in
	    -) #long flag options
		case "${OPTARG}" in
                    input_file)
                        echo "input_file : ""${!OPTIND}"        
                        INPUT_FILE="${!OPTIND}"
                        ;;
                    trained_models)
                        echo "trained_models : ""${!OPTIND}"
                        TRAINED_MODELS_PATH="${!OPTIND}"
                        ;;
                    model_nums)
                        echo "model_nums : ""${!OPTIND}"
                        MODEL_NUMS="${!OPTIND}"
                        ;;
                    num_best)
                        echo "num_best : ""${!OPTIND}"
                        KBEST_SIZE="${!OPTIND}"         
                        ;;
                    output_file)
                        echo "output_file : ""${!OPTIND}"
                        OUTPUT_FILE="${!OPTIND}"
                        ;;
                    log_file)
                        echo "log_file : ""${!OPTIND}"
                        LOG_FILE="${!OPTIND}"
                        ;;
                    extra_rnn_args)
                        >&2 echo "extra_rnn_args : ""${!OPTIND}"
                        EXTRA_RNN_ARGS="${!OPTIND}"                  
                        ;;      
                    rnn_location)
                        >&2 echo "rnn_location : ""${!OPTIND}"
                        RNN_LOCATION="${!OPTIND}"                  
                        ;;      
                    qsubopts)
                        >&2 echo "qsubopts : ""${!OPTIND}"
                        QSUBOPTS="${!OPTIND}"                  
                        ;;            
                    noqsub)
                        >&2 echo "noqsub : ""${!OPTIND}"
                        NOQSUB="${!OPTIND}"                  
                        ;;            

                    *) #unrecognized long flag
                        echo -e \\n"Option --${BOLD}$OPTARG${NORM} not allowed."
                        HELP    
                        ;;
		esac;;
	    h)  #show help
		HELP
		;;
	    \?) #unrecognized option - show help
		echo -e \\n"Option -${BOLD}$OPTARG${NORM} not allowed."
		HELP
      #If you just want to display a simple error message instead of the full
      #help, remove the 2 lines above and uncomment the 2 lines below.
      #echo -e "Use ${BOLD}$SCRIPT -h${NORM} to see the help documentation."\\n
      #exit 2
		;;
	esac
    done
    VAR=$(($VAR + 1))
#shift $((OPTIND-1))  #This tells getopts to move on to the next argument.
    shift 1
done

SPACED_MODEL_NUMS=`echo $MODEL_NUMS| sed 's/_/ /g'`;

#run some input checks
check_zero_file () {
    if [[ ! -s "$1" ]]
    then
        echo "Error file: ${BOLD}$1${NORM} is of zero size or does not exist"
        exit 1
    fi
}

check_valid_num () {
    re='^[0-9]+$'
    if ! [[ $1 =~ $re ]] ; then
        echo "Error: $1 is not a number > 0"
        exit 1
    fi
}

check_parent_structure () {
    
    if [[ -z "$1" ]]
    then
        echo "Error the flag ${BOLD}trained_models${NORM} must be specified" 
        exit 1
    fi
    
    if [[ ! -d "$1" ]]
    then
        echo "Error trained_models directory: ${BOLD}$1${NORM} does not exist"
        exit 1
    fi      
    
    for i in $SPACED_MODEL_NUMS
    do
        if [[ ! -d "$1""model""$i" ]]
        then
            echo "Error directory ${BOLD}model$i${NORM} in trained_models directory ${BOLD}$1${NORM} does not exist"
            exit 1
        fi
        
        if [[ ! -s $1"model"$i"/best.nn" ]]
        then
            echo "Error model file ${BOLD}$1"model"$i"/best.nn"${NORM} does not exist"
            exit 1
        fi
    done
}

check_dir_final_char () {
    FINAL_CHAR="${1: -1}"
    if [[ $FINAL_CHAR != "/" ]]; then
        return 1
    fi
    return 0
}

check_valid_file_path () {
    VAR=$1
    TMP_DIR=${VAR%/*}       
    if [[ ! -d $TMP_DIR ]]; then
        echo "Error path for file ${BOLD}$1${NORM} does not exist"
        exit 1
    fi
}

check_relative_path () {
    if ! [[ "$1" = /* ]]; then
        echo "Error: relative paths are not allowed for any location, ${BOLD}$1${NORM} is a relative path"
        exit 1 
    fi
}

check_berk_aligner () {
    if [[ ! -d $1"berk_aligner" ]]; then
        echo "Error: the berkeley aligner info is not in the directory ${BOLD}$1${NORM}"
        exit 1
    fi
    if [[ ! -s $1"berk_aligner/aligner_output/stage2.2.params.txt" ]]; then
        echo "Error: the ttable from the berkeley aligner at location ${BOLD}$1${NORM} is not there. It could be that the aligner needs more time to run"
        exit 1
    fi      
}

if [[ $KBEST_SIZE != 1 ]]; then
    echo "Error num_best must be equal to one in this version. An updated version is coming soon"
    exit 1  
fi


check_zero_file "$INPUT_FILE"
check_valid_num "$KBEST_SIZE"
BEAM_SIZE=$(( 12 > $KBEST_SIZE ? 12 : $KBEST_SIZE ))
check_dir_final_char "$TRAINED_MODELS_PATH"
BOOL_COND=$?
if [[ "$BOOL_COND" == 1 ]]; then
    TRAINED_MODELS_PATH=$TRAINED_MODELS_PATH"/"
fi
check_parent_structure "$TRAINED_MODELS_PATH"
LONGEST_SENT=$( wc -L $INPUT_FILE | cut -f1 -d' ' )
check_valid_file_path "$OUTPUT_FILE"
check_valid_file_path "$LOG_FILE"
check_relative_path "$INPUT_FILE"
check_relative_path "$TRAINED_MODELS_PATH"
check_relative_path "$OUTPUT_FILE"
check_relative_path "$LOG_FILE"
check_berk_aligner "$TRAINED_MODELS_PATH"

#paths
BLEU_FORMAT="${DIR}helper_programs/bleu_format.py"
UNK_REP="${DIR}helper_programs/att_unk_rep.py"
DECODE_FORMAT="${DIR}helper_programs/decode_format.py"
TTABLE="${TRAINED_MODELS_PATH}berk_aligner/aligner_output/stage2.2.params.txt"
MODEL_NAMES=""

MFLAGS="-M";
nextgpu=0
for i in $SPACED_MODEL_NUMS; do    
    CURR_MODEL_NAME="${TRAINED_MODELS_PATH}model${i}/best.nn"
    check_zero_file $CURR_MODEL_NAME;
    MODEL_NAMES=$MODEL_NAMES" $CURR_MODEL_NAME";
    MFLAGS="$MFLAGS $nextgpu";
    nextgpu=$(((nextgpu+1)%2)) # flip between 0 and 1
    #HACKY FIX FOR OLD CODE
    TEMP_FIRST=$( head -1 $CURR_MODEL_NAME )
    TEMP_ARR=($TEMP_FIRST)
    if [[ ${#TEMP_ARR[@]} < 5 ]]; then
        TEMP_FIRST=$TEMP_FIRST" 1 1 0 0 0"
        sed -i "1s/.*/$TEMP_FIRST/" $CURR_MODEL_NAME 
    fi
done
EXTRA_RNN_ARGS=`echo $EXTRA_RNN_ARGS | sed 's/__/--/g'`;
FINAL_ARGS="\" $RNN_LOCATION --logfile $LOG_FILE -k $KBEST_SIZE $MODEL_NAMES $OUTPUT_FILE -b $BEAM_SIZE --print-score 1 $MFLAGS -L $LONGEST_SENT $EXTRA_RNN_ARGS \""
DECODE_SCRIPT="${DIR}helper_programs/decode_single.sh"
if [[ $NOQSUB == 0 ]]; then
    SMART_QSUB="${DIR}helper_programs/qsubrun"
    QSUBOPTS=`echo $QSUBOPTS | sed 's/_/-/g'`;
    QSUB="$SMART_QSUB";
    if [ ! -z "$QSUBOPTS" ]; then
	QSUB="$QSUB $QSUBOPTS -- ";
    fi
    $QSUB $DECODE_SCRIPT $FINAL_ARGS $INPUT_FILE $BLEU_FORMAT $OUTPUT_FILE $TTABLE $UNK_REP $DECODE_FORMAT $MODEL_NUMS;
else
    cmd="$DECODE_SCRIPT $FINAL_ARGS $INPUT_FILE $BLEU_FORMAT $OUTPUT_FILE $TTABLE $UNK_REP $DECODE_FORMAT $MODEL_NUMS";
    echo $cmd;
    exec $cmd;
fi

exit $?

