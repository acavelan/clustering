TARGET_NAME=report

# all requiert la table des matière et la bibliographie
all: 
#$(TARGET_NAME).toc $(TARGET_NAME).bbl
	pdflatex -shell-escape -halt-on-error -file-line-error -halt-on-error $(TARGET_NAME).tex

# Compile tout de A à Z
complete: cleanall all

$(TARGET_NAME).toc: $(TARGET_NAME).aux

$(TARGET_NAME).bbl: $(TARGET_NAME).aux
	bibtex $(TARGET_NAME)

$(TARGET_NAME).aux:
	pdflatex -shell-escape -draftmode -halt-on-error -file-line-error $(TARGET_NAME).tex

.PHONY: clean cleanall

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.pyg *.xml *.blg *.bbl *.bcf $(TARGET_NAME)-blx.bib

cleanall: clean
	rm -f $(TARGET_NAME).pdf

