%% Extraction of relevant data from Hofree et al. dataset
% You need to download code and data from:
% http://chianti.ucsd.edu/~mhofree/wordpress/?page_id=26
% Put then the script at the root folder "nbs_release_v0.2"
% The two files 'adj_mat.mat' and 'entrez_to_idmat.mat' should then be copied in the "stratipy/data" folder with the original mutation data file "somatic_data_UCEC.mat" and phenotype data file "UCEC_clinical_phenotype.mat" (both localized in the "nbs_release_v0.2/data/TCGA_somatic_mutations folder").

library_path = pwd;
addpath(genpath(library_path))
basedata_path = [library_path '/data'];

NBSnetwork = load([ basedata_path '/networks/ST90Q_adj_mat.mat' ]);
adj_mat=NBSnetwork.network.adj_mat;
save([ basedata_path 'adj_mat','adj_mat'])

entrezid={}
keymat={}
bin=1;
for k = NBSnetwork.network.key
    try
        entrezid{bin}=NBSnetwork.network.KeytoEntrezIDMap(k{1});
    catch
        entrezid{bin}=NaN;
    end
    disp([k{1},',',num2str(entrezid{bin})])
    keymat{bin}=k{1};
    bin=bin+1;
end
save([ basedata_path 'entrez_to_idmat'],'entrezid','keymat')