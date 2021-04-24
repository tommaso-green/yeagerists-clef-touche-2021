package it.unipd.dei.yeagerists.search;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class ResultDocument {
    private String id;
    private String stance;
    private String body;
    private float score;
    private String queryId;
}
