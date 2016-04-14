#!/bin/bash
#PBS -q isi
#PBS -l walltime=336:00:00
#PBS -l gpus=2

#This script was written by barret zoph for questions email barretzoph@gmail.com

#### Things that must be specified by user ####
MODEL_DIR="" 
OUTPUT_DIR="" 

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )""/"

NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo "${BOLD}The following switches are required:${NORM}"
  echo "${REV}--model${NORM}  : Specify the location of the trained NMT models."
  echo "${REV}--output${NORM}  : Specify the location where all the output graphs and data will be put."
  echo -e "${REV}-h${NORM}  : Displays this help message. No further functions are performed."\\n
  echo "Example: ${BOLD}$SCRIPT --model <trained_model> --source <rescore_source> --target <rescore_target> --scores <rescore_scores> ${NORM}" 
  exit 1
}

#Check the number of arguements. If none are passed, print help and exit.
NUMARGS=$#

if [[ $NUMARGS != "4" ]]; then
    echo -e \\n"Number of arguments: $NUMARGS"
    HELP
fi

VAR="1"
optspec=":h-:"
NUM=$(( 1 > $(($NUMARGS/2)) ? 1 : $(($NUMARGS/2)) ))
while [[ $VAR -le $NUM ]]; do
while getopts $optspec FLAG; do
  case $FLAG in
    -) #long flag options
	case "${OPTARG}" in
		model)
			echo "model = ""${!OPTIND}"	
			MODEL_DIR="${!OPTIND}"
			;;
		output)
			echo "output = ""${!OPTIND}"
			OUTPUT_DIR="${!OPTIND}"
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
shift 1  #This tells getopts to move on to the next argument.
done
### End getopts code ###

check_zero_dir () {
	if [[ ! -d "$1" ]]
	then
		echo "Error directory: ${BOLD}$1${NORM} does not exist"
		exit 1
	fi
}

check_dir_final_char () {
	FINAL_CHAR="${1: -1}"
	if [[ $FINAL_CHAR != "/" ]]; then
		return 1
	fi
	return 0
}

check_parent_structure () {
	
	if [[ -z "$1" ]]
	then
		echo "Error the flag ${BOLD}parent_model${NORM} must be specified" 
		exit 1
	fi
	
	if [[ ! -d "$1" ]]
	then
		echo "Error parent model directory: ${BOLD}$1${NORM} does not exist"
		exit 1
	fi	
	
	for i in $( seq 1 8 )
	do
		if [[ ! -d "$1""model""$i" ]]
		then
			echo "Error directory ${BOLD}model$i${NORM} in parent model directory ${BOLD}$1${NORM} does not exist"
			exit 1
		fi
		
		if [[ ! -s $1"model"$i"/best.nn" ]]
		then
			echo "Error model file ${BOLD}$1"model"$i"/best.nn"${NORM} does not exist"
			exit 1
		fi
	done
}

check_zero_dir "$OUTPUT_DIR"
check_dir_final_char "$MODEL_DIR"
check_dir_final_char "$OUTPUT_DIR"
check_parent_structure "$MODEL_DIR"
touch $OUTPUT_DIR"info.txt"
for i in $( seq 1 8 ); do
	python $DIR"helper_programs/make_graph.py" $MODEL_DIR"model${i}/HPC_OUTPUT.txt" $i $OUTPUT_DIR $OUTPUT_DIR"info.txt"
	 
done


