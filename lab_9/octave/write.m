function write(filename, betas, ws)
    file = fopen(filename, "w");
    for beta = betas
        fprintf(file, "%g ", beta);
    endfor
    for w = ws
        fprintf(file, "\n%g", w);
    endfor
    fclose(file);
endfunction