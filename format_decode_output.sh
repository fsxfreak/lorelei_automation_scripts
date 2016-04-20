#!/bin/bash
#PBS -q isi
#PBS -l walltime=336:00:00

source /usr/usc/python/2.7.8/setup.sh

#Set Script Name variable
SCRIPT=`basename ${BASH_SOURCE[0]}`
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )""/"

tmpdir=${TMPDIR:-/tmp}
MTMP=$(mktemp -d --tmpdir=$tmpdir XXXXXX)
function cleanup() {
    rm -rf $MTMP;
}
trap cleanup EXIT

#Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

#Help function
function HELP {
    echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n 
    echo "${BOLD}The following switches are:${NORM}"
    echo "${REV}--xml_file${NORM}  : Specify the location of the xml file that is used for formatting." 
    echo "${REV}--nmt_output${NORM}  : Specify the location of the NMT output that has been decoded." 
    echo "${REV}--output_name${NORM}  : Specify the name and location of the output file in the correct format. The final output file will be named \"output_name\".xml.gz" 
    echo -e "${REV}-h${NORM}  : Displays this help message. No further functions are performed."\\n
    echo "Example: ${BOLD}$SCRIPT path/to/format_decode_output.sh --xml_file <xml file> --nmt_output <nmt output> --output_name <output name> ${NORM}" 
    exit 1
}

XML_FILE=""
NMT_OUTPUT=""
OUTPUT_NAME=""

#Check the number of arguments. If none are passed, print help and exit.
NUMARGS=$#

echo "Number of arguements specified: ${NUMARGS}"

if [[ $NUMARGS -lt 6 ]]; then
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
                    xml_file)
                        echo "xml_file : ""${!OPTIND}"        
                        XML_FILE="${!OPTIND}"
                        ;;
                    nmt_output)
                        echo "nmt_output : ""${!OPTIND}"
                        NMT_OUTPUT="${!OPTIND}"
                        ;;
                    output_name)
                        echo "output_name : ""${!OPTIND}"
                        OUTPUT_NAME="${!OPTIND}"
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

TAB_OUTPUT="${DIR}helper_programs/tab_output.py"
XMLIFY="${DIR}helper_programs/xmlify-nbest"
DETOK="${DIR}helper_programs/lw_detokenize.pl"

cd $tmpdir
cp $XML_FILE output.xml
cat $NMT_OUTPUT | cut -d$'\t' -f4 > sent_only.txt
perl $DETOK < sent_only.txt > sent_only.txt.detok
python $TAB_OUTPUT sent_only.txt.detok
cat sent_only.txt.detok.tab | python $XMLIFY output.xml | gzip > ${OUTPUT_NAME}".xml.gz"


