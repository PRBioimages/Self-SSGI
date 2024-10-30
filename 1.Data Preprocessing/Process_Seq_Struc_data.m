clc
clear all
path = '.\PDB_Files';
files_and_folders = dir(path);
files = files_and_folders(~[files_and_folders.isdir] & ~ismember({files_and_folders.name}, {'.', '..'}));
full_paths = fullfile(path, {files.name});

for i=1:length(full_paths)
    disp(i) 
    pdb= pdbread (full_paths{i}); 
    seq_length=pdb.Sequence.NumOfResidues;
    all_fea=zeros(seq_length,6);
    amino_acids = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'};
    
    %%Get dihedral features
    Rall = ramachandran(pdb, 'plot', 'none');
    R = Rall(1);     
    for j=2:numel(Rall)
       if ~ismember(Rall(j).Chain,R.Chain)              
            R.Angles = [R.Angles; Rall(j).Angles];
            R.ResidueNum = [R.ResidueNum; Rall(j).ResidueNum];
            R.ResidueName = [R.ResidueName; Rall(j).ResidueName];
            R.Chain = [R.Chain; Rall(j).Chain];   
       end
    end
    R.Angles(isnan(R.Angles)) = 0;
    if (length(R.Angles)~=seq_length)
        disp('error')
    end
    count=0;
    for j=1:length(pdb.Model.Atom)
        if strcmp(pdb.Model.Atom(j).AtomName, 'CA')
            count=count+1;
            %%Obtain amino acid type information and three-dimensional coordinates
            all_fea(count,1)=R.Angles(count,1);
            all_fea(count,2)=R.Angles(count,2);
            all_fea(count,3)=pdb.Model.Atom(j).X;
            all_fea(count,4)=pdb.Model.Atom(j).Y;
            all_fea(count,5)=pdb.Model.Atom(j).Z;

            res_name=pdb.Model.Atom(j).resName;
            one_hot_encoding = zeros(1, length(amino_acids));
            index = find(strcmp(amino_acids, res_name));
            one_hot_encoding(index) = 1;
            %all_fea(count,6:25)=one_hot_encoding;
            all_fea(count,6)=find(one_hot_encoding == 1)-1;%-1because Python uses 0-based indexing, so the first element is index 0.
        end
        
    end
    name=['.\Seq_Struc_Fea\Types_Coordinates_Angles\',files(i).name,'.mat'];
    save(name,'all_fea') 
end


