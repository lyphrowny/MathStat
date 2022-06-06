dir = "../data/";
files = {"ch1_800nm_0.04.csv"; "ch2_800nm_0.04.csv"};
dirs = cellstr(zeros(length(files), 1) + dir);

files = strcat(dirs, files);
tol = 1e-4;

for i = 1:length(files)
    file = char(files{i});
    data = csvread(file);

    enum = transpose(1:length(data));
    # col of ones; col of natural seq
    A = [enum.^0, enum];
    inf = data - tol;
    sup = data + tol;

    [betas, ws] = min(A, inf, sup);
    # change the extension of the file
    write(strrep(file, "csv", "txt"), betas, ws);
endfor
