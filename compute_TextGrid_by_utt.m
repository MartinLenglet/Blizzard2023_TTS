% align_csv = 'save_Blizzard2023/NEB_train.csv';
align_csv = 'save_Blizzard2023/AD_train.csv';

output_file = 'seg_by_utt_Blizzard2023';

A = read_csv(align_csv, 6);
nbr_utt = length(A{1});

skip_until = '';
skip = false;

name_previous_chapter = '';
for i_utt = 1:nbr_utt
    name_current_chapter = A{1}{i_utt};
    split_name_current_chapter = split(name_current_chapter, '_');
    if length(split_name_current_chapter) >= 6
        current_speaker = split_name_current_chapter{4};
    else
        current_speaker = split_name_current_chapter{3};
    end
    
    if ~strcmp(skip_until, '') && ~strcmp(name_current_chapter, skip_until) && skip
        continue;
    else
        skip = false;
    end
   
    current_text = A{4}{i_utt};
    split_current_text = phon_input_to_cell_array(current_text);
    current_phon_align = A{5}{i_utt};
    split_current_phon_align = split(current_phon_align, ' ');
    current_duration = A{6}{i_utt};
    split_current_duration = str2double(split(current_duration, ' '))/1000;
    if (strcmp(name_current_chapter, name_previous_chapter))
        nbr_utt_in_chapter = nbr_utt_in_chapter + 1;
    else
        nbr_utt_in_chapter = 1;
    end
    name_previous_chapter = name_current_chapter;
    
    fprintf('%s_%d | %d/%d\n', name_current_chapter, nbr_utt_in_chapter, i_utt, nbr_utt);

    % Init list
    start_time = 0;
    end_time = 0;
    x_max = sum(split_current_duration);
    nbr_char = length(split_current_text);

    % Write TextGrid header
    current_utt_textgrid = fopen(sprintf('%s/%s/%s_%d.TextGrid', output_file, current_speaker, name_current_chapter, nbr_utt_in_chapter), 'w');
    fprintf(current_utt_textgrid, 'File type = "ooTextFile"\n');
    fprintf(current_utt_textgrid, 'Object class = "TextGrid"\n');
    fprintf(current_utt_textgrid, "\n");
    fprintf(current_utt_textgrid, "xmin = 0\n");
    fprintf(current_utt_textgrid, sprintf("xmax = %.3f\n", x_max));
    fprintf(current_utt_textgrid, "tiers? <exists>\n");
    fprintf(current_utt_textgrid, "size = 1\n");
    fprintf(current_utt_textgrid, "item []:\n");
    fprintf(current_utt_textgrid, "\titem [1]:\n");
    fprintf(current_utt_textgrid, '\t\tclass = "IntervalTier"\n');
    fprintf(current_utt_textgrid, '\t\tname = "phones"\n');
    fprintf(current_utt_textgrid, "\t\txmin = 0\n");
    fprintf(current_utt_textgrid, sprintf("\t\txmax = %.3f\n", x_max));
    
    fprintf(current_utt_textgrid, sprintf("\t\tintervals: size = %d\n", nbr_char));
    
    for i_char = 1:nbr_char
        end_time = end_time + split_current_duration(i_char);
        
        current_char = split_current_text{i_char};
        
        % TextGrid
        fprintf(current_utt_textgrid, sprintf("\t\tintervals [%d]:\n", i_char));
        fprintf(current_utt_textgrid, sprintf("\t\t\txmin = %.3f\n", start_time));
        fprintf(current_utt_textgrid, sprintf("\t\t\txmax = %.3f\n", end_time));
        fprintf(current_utt_textgrid, sprintf('\t\t\ttext = "%s"\n', current_char));
        
        start_time = start_time + split_current_duration(i_char);
    end
    
    fclose(current_utt_textgrid);
end