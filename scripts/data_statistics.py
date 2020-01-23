import pandas as pd

def AP_PA(master):
    master['Study'] = master['StudyFile'].apply(lambda x: "/".join(x.split("/")[:-1]))
    master['Patient'] = master['StudyFile'].apply(lambda x: x.split("/")[4])
    dataset_studies = master[master['DataSplit'].isin(['train', 'valid', 'test'])]

    num_images = dataset_studies.shape[0]
    num_patients = len(dataset_studies['Patient'].unique())
    num_studies = len(dataset_studies['Study'].unique())
    num_frontal = dataset_studies[dataset_studies["Frontal/Lateral"] == "Frontal"].shape[0]
    percent_frontal = num_frontal / num_images
    num_lateral = dataset_studies[dataset_studies["Frontal/Lateral"] == "Lateral"].shape[0]
    percent_lateral = num_lateral / num_images
    num_ap = dataset_studies[dataset_studies["AP/PA"] == "AP"].shape[0]
    percent_ap = num_ap / num_frontal
    num_pa = dataset_studies[dataset_studies["AP/PA"] == "PA"].shape[0]
    percent_pa = num_pa / num_frontal

    print("Number of images: {}".format(num_images))
    print("Number of patients: {}".format(num_patients))
    print("Number of studies: {}".format(num_studies))
    print("Percent of Frontal: {}".format(percent_frontal))
    print("Percent of Lateral: {}".format(percent_lateral))
    print("Percent of Frontal that are AP: {}".format(percent_ap))
    print("Percent of Frontal that are PA: {}".format(percent_pa))


def labeler_prevalence(master):
    master['Study'] = master['StudyFile'].apply(lambda x: "/".join(x.split("/")[:-1]))
    categories = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
                  'Airspace Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    relevant_study_rows = master[master['DataSplit'].isin(['train'])][['Study'] + categories].drop_duplicates()

    positive_prevalences = (relevant_study_rows[categories] == 1).sum(axis=0)
    uncertain_prevalences = (relevant_study_rows[categories] == -1).sum(axis=0)
    negative_prevalences = (relevant_study_rows[categories] == 0).sum(axis=0)

    all_prevalences = pd.concat([positive_prevalences, uncertain_prevalences, negative_prevalences], axis=1)
    total = all_prevalences.sum(axis=1)

    for i in range(3):
        percents = (all_prevalences[i] * 100 / total).round(2)
        all_prevalences[i] = all_prevalences[i].map(str) + ' (' + percents.map(str) + ')'
    print(all_prevalences)
    print(all_prevalences.to_latex())



if __name__ == "__main__":

    master = pd.read_csv("/data3/xray4all/master_unpostprocessed.csv")
    labeler_prevalence(master)


