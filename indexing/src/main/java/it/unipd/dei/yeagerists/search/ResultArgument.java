package it.unipd.dei.yeagerists.search;

import lombok.Builder;
import lombok.Data;

/**
 * Contains data of an argument read from the index.
 */
@Data
@Builder
public class ResultArgument {
    private String id;
    private String stance;
    private String body;
    private String title;
    private float score;
    private String queryId;
}
